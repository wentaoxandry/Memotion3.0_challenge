import os, sys, json
import random
from model import *
from utils import *
import collections
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import csv
SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def training(config, dataset=None):
    criterion = torch.nn.CrossEntropyLoss()
    model = CLIPtext(config["MODELtext"], config["cachedir"], config["dropout"])
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    evalacc_best = 0
    early_wait = 9
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    #loaded_state = torch.load(config["pretraineddir"], map_location=config["device"]).state_dict()
    loaded_state = torch.load(config["pretraineddir"], map_location=config["device"])[0]
    self_state = model.state_dict()

    loaded_state = {k: v for k, v in loaded_state.items() if not k.startswith('output')}
    self_state.update(loaded_state)
    model.load_state_dict(self_state)
    #cdcm
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(config['device'])


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  eps=config["eps"], weight_decay=config["weight_decay"]
                                  )
    train_examples_len = len(dataset.train_dataset)
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))
    '''scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 1,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])'''
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=0, verbose=True)
    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_clip_custom_sequence)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, drop_last=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_clip_custom_sequence)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(data_loader_train, 0):
            node_sets = data[0]
            mask = data[1]
            label = data[2].to(config["device"])
            filename = data[3]
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask)
            label = label.squeeze(-1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            print("\r%f" % loss, end='')

            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().detach().data.numpy().tolist())
            del loss, outputs, node_sets, mask, label
        trainallscore = compute_f1(trainpredict, trainlabel)
        # np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(
        # trainlabel)

        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        evallossvec = []
        evalacc = 0
        model.eval()
        evalpred = []
        evallabel = []
        outpre = {}
        total = 0
        for i, data in enumerate(data_loader_dev, 0):
            node_sets = data[0]
            mask = data[1]
            labels = data[2].to(config["device"])
            labels = labels.squeeze(-1)
            filename = data[3]
            outputs = model(node_sets, mask)
            dev_loss = criterion(outputs, labels)
            evallossvec.append(dev_loss.cpu().data.numpy())
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            evalpred.extend(predicted.cpu().detach().tolist())
            evallabel.extend(labels.cpu().detach().data.numpy().tolist())

            total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filename[i]].update(
                    {'prob': torch.softmax(outputs[i], dim=-1).cpu().detach().data.numpy().tolist()})
        del dev_loss, outputs, node_sets, mask, labels


        allscore = compute_f1(evalpred, evallabel)  # correct / total
        #scheduler.step(allscore)
        # evalacc = evalacc / len(evallabel)
        evallossmean = np.mean(np.array(evallossvec))
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,
                                  str(epoch) + '_' + str(evallossmean) + '_' + str(
                                      currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                      allscore)[:6] + '.pkl')
        torch.save(model, OUTPUT_DIR)
        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
            allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)

        torch.cuda.empty_cache()
        if allscore < evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break


def main(datadir, savedir, cashedir, bestmodelinfo, pretraininfo):
    res = tuple(bestmodelinfo.split(', '))
    lr = float(res[1].strip(')'))
    dropout = float(res[2].strip(')'))
    weight_decay = float(res[3].strip(')'))
    pretrainres = tuple(pretraininfo.split(', '))
    pretraineddir = pretrainres[-1].strip(')').strip('\'')#.replace('./', '/Memotion3/')
    max_num_epochs = 70
    print(pretraineddir)

    ## 4-folder cross-validation
    modeldir = os.path.join(savedir, 'text', 'model')
    resultsdir = os.path.join(savedir, 'text', 'results')

    for makedir in [modeldir, resultsdir, cashedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "memotion3", "train_en.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "memotion3", "val_en.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    tokenizer = CLIPTokenizer.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = BERTweetdatasetclass(train_file=traindict,
                                   test_file=valdict,
                                   tokenizer=tokenizer,
                                   device=device,
                                   max_len=max_len)

    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "weight_decay": weight_decay,
        "eps": 1e-8,
        "lr": lr,
        "dropout": dropout,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "batch_size": 48,  # 16,
        "cachedir": cashedir,
        "epochs": max_num_epochs,
        "pretraineddir": pretraineddir
    }
    training(config, dataset)










'''datadir = './../../../dataset'
outdir = './../../../output/text/taskA/train'
cashedir = './../../../CASHE'
main(datadir, outdir, cashedir)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


