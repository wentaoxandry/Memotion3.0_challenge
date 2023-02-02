import os, sys, json
import random
from model import *
from utils import *
import collections
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
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
    criterion = FocalLoss(gamma=config["gamma"])  #torch.nn.CrossEntropyLoss()
    model = CLIPmulti(config["MODELtext"], config["cachedir"], config["dropout"], config["pretrainedtextdir"], config["pretrainedimagedir"], type=config["type"])
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    evalacc_best = 0
    early_wait = 4
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(config['device'])

    '''if config["pretrainedtextdir"] == None:
        pass
    else:
        loaded_state = torch.load(config["pretrainedtextdir"], map_location=config["device"])
        if isinstance(loaded_state, torch.nn.DataParallel):
            loaded_state = loaded_state.module
        loaded_state = loaded_state.state_dict()
        self_state = model.state_dict()

        loaded_state = {k.replace('cliptext.', 'clipmulti.'): v for k, v in loaded_state.items() if
                        k.replace('cliptext.', 'clipmulti.') in self_state and not k.replace('cliptext.',
                                                                                             'clipmulti.').startswith(
                            'linear')}
        self_state.update(loaded_state)
        model.load_state_dict(self_state)

        loaded_state = torch.load(config["pretrainedimagedir"], map_location=config["device"])
        if isinstance(loaded_state, torch.nn.DataParallel):
            loaded_state = loaded_state.module
        loaded_state = loaded_state.state_dict()
        self_state = model.state_dict()

        loaded_state = {k.replace('clipimage.', 'clipmulti.'): v for k, v in loaded_state.items() if
                        k.replace('clipimage.', 'clipmulti.') in self_state and not k.replace('cliptext.',
                                                                                              'clipmulti.').startswith(
                            'linear')}
        self_state.update(loaded_state)
        model.load_state_dict(self_state)'''



    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  eps=config["eps"], weight_decay=config["weight_decay"]
                                  )
    train_examples_len = len(dataset.train_dataset)
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 5,
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


    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(data_loader_train, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            label = data[3].to(config["device"])
            filename = data[4]
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask, image)
            label = label.squeeze(-1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            print("\r%f" % loss, end='')

            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainpredict.extend(one_hot(predicted.cpu().detach().data.numpy(), outputs.size(1)).tolist())
            trainlabel.extend(one_hot(label.cpu().data.numpy(), outputs.size(1)).tolist())
            #del loss, outputs, node_sets, mask, label
        #trainallscore = np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(
            #trainlabel)
        cates = outputs.size(1)
        results = []
        total_occurences = 0
        for index in range(cates):
            label = []
            predict = []
            for i in range(len(trainlabel)):
                label.extend([trainlabel[i][index]])
                predict.extend([trainpredict[i][index]])
            f1_score = compute_f1(predict, label)
            f1weight = label.count(True)
            total_occurences += f1weight
            results.append(f1_score * f1weight)
        trainallscore = sum(results) / total_occurences

        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        evallossvec = []
        evalacc = 0
        model.eval()
        correct = 0
        outpre = {}
        total = 0
        evalpredict = []
        evallabel = []
        for i, data in enumerate(data_loader_dev, 0):
            with torch.no_grad():
                node_sets = data[0].to(config["device"])
                mask = data[1].to(config["device"])
                image = data[2].to(config["device"])
                label = data[3].to(config["device"])
                labels = label.squeeze(-1)
                outputs = model(node_sets, mask, image)
                dev_loss = criterion(outputs, labels)
                evallossvec.append(dev_loss.cpu().data.numpy())

                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for i in range(len(filename)):
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                    outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                evalpredict.extend(one_hot(predicted.cpu().detach().data.numpy(), outputs.size(1)).tolist())
                evallabel.extend(one_hot(labels.cpu().data.numpy(), outputs.size(1)).tolist())
            #del dev_loss, outputs, node_sets, mask, labels

        allscore = correct / total
        cates = outputs.size(1)
        results = []
        total_occurences = 0
        for index in range(cates):
            label = []
            predict = []
            for i in range(len(evallabel)):
                label.extend([evallabel[i][index]])
                predict.extend([evalpredict[i][index]])
            f1_score = compute_f1(predict, label)
            f1weight = label.count(True)
            total_occurences += f1weight
            results.append(f1_score * f1weight)
        allscore = sum(results) / total_occurences
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


def main(datadir, savedir, cashedir, type, bestmodelinfo):
    res = tuple(bestmodelinfo.split(', '))
    lr = float(res[1])
    dropout = float(res[2])
    weight_decay = float(res[3])
    gamma = int(res[5])
    raycheckdir = res[6].strip(')')
    npdatadir = os.path.join(datadir, 'pretrainCLIP')
    max_num_epochs = 20
    '''pretraineddir = os.path.join('/home/wentao/Desktop/Memotion3/output', 'taskA', 'pretrain',
                                 'text', 'model',
                                 '1_0.67794627_8e-05_0.7235_0.6227.pkl')'''
    #pretraineddir = os.path.join('./output', 'taskA', 'pretrain',
    #                             'text', 'model',
    #                             '1_0.67794627_8e-05_0.7235_0.6227.pkl')

    ## 4-folder cross-validation
    for cv in range(4):
        best_model = 'acc'
        pretraineddirs = []
        for mode in ['text', 'image']:
            modeldir = os.path.join(savedir, mode, type, str(cv),
                                    'model')
            netlist = os.listdir(modeldir)
            netlist.sort()
            net_dict = {}
            if best_model == 'loss':
                for m in range(len(netlist)):
                    templist = netlist[m].split('_')
                    net_dict[templist[1]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items(), reverse=True))
            else:
                for m in range(len(netlist)):
                    templist = netlist[m].split('_')
                    net_dict[templist[4]] = netlist[m]
                    net_dict = collections.OrderedDict(sorted(net_dict.items()))
            netname = net_dict.get(list(net_dict)[-1])
            pretraineddirs.append(os.path.join(modeldir,
                                               netname))
        modeldir = os.path.join(savedir, 'multi', type, str(cv), 'model')
        resultsdir = os.path.join(savedir, 'multi', type, str(cv), 'results')

        for makedir in [modeldir, resultsdir, cashedir]:
            if not os.path.exists(makedir):
                os.makedirs(makedir)

        if torch.cuda.is_available() == True:
            device = 'cuda'
        else:
            device = 'cpu'

        with open(os.path.join(datadir, "memotion3", "train.json"), encoding="utf8") as json_file:
            traindict = json.load(json_file)

        ## load dev features
        with open(os.path.join(datadir, "memotion3", 'cvtest' + str(cv) + '.txt')) as f:
            lines = f.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].strip('\n')
        newtraindict = {}
        newtestdict = {}
        for i in list(traindict.keys()):
            if i in lines:
                newtestdict.update({i: traindict[i]})
            else:
                newtraindict.update({i: traindict[i]})

        MODELtext = "openai/clip-vit-base-patch32"

        dataset = BERTweetdatasetclass(train_file=newtraindict,
                                       test_file=newtestdict,
                                       device=device,
                                       npdatadir=npdatadir,
                                       type=type)

        config = {
            "MODELtext": MODELtext,
            "NWORKER": 0,
            "device": device,
            "weight_decay": weight_decay,
            "eps": 1e-8,
            "lr": lr,
            "modeldir": modeldir,
            "resultsdir": resultsdir,
            "dropout": dropout,
            "batch_size": 48,  # 16,
            "cachedir": cashedir,
            "epochs": max_num_epochs,
            "pretrainedtextdir": pretraineddirs[0],
            "pretrainedimagedir": pretraineddirs[1],
            "type": type,
            "gamma": gamma
        }
        training(config, dataset)








'''datadir = './../../../dataset'
outdir = './../../../output/taskC/train'
cashedir = './../../../CASHE'
type = 'motivation'
main(datadir, outdir, cashedir, type)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


