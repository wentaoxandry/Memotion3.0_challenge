import os, sys, json
import random
from model import CLIP
from utils import *
import numpy as np
from transformers import get_linear_schedule_with_warmup
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
    model = CLIP(config["MODELtext"], config["cachedir"])
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    evalacc_best = 0
    evalloss_best = np.Inf
    early_wait = 4
    run_wait = 1
    continuescore = 0
    stop_counter = 0
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
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 1,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])
    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_clip_custom_sequence)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, drop_last=True,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_clip_custom_sequence)

    allbatch = int(train_examples_len / config["batch_size"])
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        for i, data in enumerate(data_loader_train, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            # zero the parameter gradients
            optimizer.zero_grad()
            loss = model(node_sets, mask, pixel)
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

            if i % 500 == 0:
                print("\r%f" % loss.mean(), end='\n')
                print(str(i) + ' / ' + str(allbatch))
            #for param_group in optimizer.param_groups:
            #  currentlr = param_group['lr']
            #print(float(currentlr))

            # print statistics
            tr_loss += loss.mean().item()
            nb_tr_steps += 1
            del loss, node_sets, mask

        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        evallossvec = []
        model.eval()
        for i, data in enumerate(data_loader_dev, 0):
            with torch.no_grad():
                node_sets = data[0].to(config["device"])
                mask = data[1].to(config["device"])
                pixel = data[2].to(config["device"])
                dev_loss = model(node_sets, mask, pixel)
                dev_loss = dev_loss.mean()
                evallossvec.append(dev_loss.cpu().data.numpy())

            del dev_loss, node_sets, mask


        # evalacc = evalacc / len(evallabel)
        evallossmean = np.mean(np.array(evallossvec))
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,
                                  str(epoch) + '_' + str(evallossmean) + '_' + str(
                                      currentlr) + '.pkl')
        torch.save(model, OUTPUT_DIR)


        torch.cuda.empty_cache()
        if evallossmean > evalloss_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalloss_best = evallossmean
            continuescore = continuescore + 1

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break


def main(datadir, savedir, cashedir, bestmodelinfo):
    res = tuple(bestmodelinfo.split(', '))
    lr = float(res[1])
    weight_decay = float(res[2])
    npdatadir = os.path.join(datadir, 'pretrainCLIP')
    max_num_epochs = 40

    modeldir = os.path.join(savedir, 'CLIP', 'model')
    resultsdir = os.path.join(savedir, 'CLIP', 'results')

    for makedir in [modeldir, resultsdir, cashedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

################################################## change




    with open(os.path.join(datadir, "pretrain_CLIP_train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)

    ## load dev features
    with open(os.path.join(datadir, "pretrain_CLIP_test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    config = {
        "MODELtext" :MODELtext,
        "NWORKER": 0,
        "device": device,
        "weight_decay": weight_decay,
        "eps": 1e-8,
        "lr": lr,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "batch_size": 16, #16,
        "cachedir": cashedir,
        "epochs": max_num_epochs
    }

    dataset = CLIPdatasetclass(train_file=traindict,
                               test_file=testdict,
                               device=device,
                               max_len=max_len,
                               npdatadir=npdatadir)


    training(config, dataset)


'''datadir = './../../dataset'
outdir = './../../output/text/taskA/pretrain'
cashedir = './../../CASHE'
main(datadir, outdir, cashedir)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


