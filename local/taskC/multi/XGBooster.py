import os, sys, json
import random
from model import *
from utils import *
import numpy as np
import xgboost as xgb
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from scipy.special import softmax
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
    criterion = FocalLoss(gamma=0)  #torch.nn.CrossEntropyLoss()
    model = CLIPmulti(config["MODELtext"], config["cachedir"], type=config["type"])
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

    if config["pretraineddir"] == None:
        pass
    else:
        loaded_state = torch.load(config["pretraineddir"], map_location=config["device"]).state_dict()
        self_state = model.state_dict()

        loaded_state = {k: v for k, v in loaded_state.items() if not k.startswith('linear.')}
        self_state.update(loaded_state)
        model.load_state_dict(self_state)



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
            trainpredict.extend(predicted.cpu().detach().data.numpy().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
            #del loss, outputs, node_sets, mask, label
        #trainallscore = np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(
            #trainlabel)
        trainallscore = f1_score(trainlabel, trainpredict, average='weighted')
        '''cates = outputs.size(1)
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
        trainallscore = sum(results) / total_occurences'''

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
                evalpredict.extend(predicted.cpu().detach().data.numpy().tolist())
                evallabel.extend(labels.cpu().data.numpy().tolist())
            #del dev_loss, outputs, node_sets, mask, labels

        '''allscore = correct / total
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
        allscore = sum(results) / total_occurences'''

        allscore = f1_score(evallabel, evalpredict, average='weighted')
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


def main(datadir, savedir, cashedir, type):
    if type != 'motivation':
        n_class = 4
    else:
        n_class = 2

    ## 4-folder cross-validation
    modeldir = os.path.join(savedir, 'XGBooster', type)
    resultsdir = os.path.join(savedir, 'XGBooster', type)

    for makedir in [modeldir, resultsdir, cashedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    with open(os.path.join(datadir, "memotion3", "train_en.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "memotion3", "val_en.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)
    with open(os.path.join(datadir, "memotion3", "test_en.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    evalX = []
    evalY = []
    filename = []
    for testkey in list(valdict.keys()):
        with open(os.path.join(datadir, 'XGBoosterfeat/taskC', type, 'val', testkey + '.npy'), 'rb') as f:
            evalX.append(np.load(f))
            evalY.append(np.load(f))
            filename.append(testkey)
    evalX = np.asarray(evalX)
    evalY = np.asarray(evalY)

    testX = []
    filename = []
    for testkey in list(testdict.keys()):
        with open(os.path.join(datadir, 'XGBoosterfeat/taskC', type, 'test', testkey + '.npy'), 'rb') as f:
            testX.append(np.load(f))
            filename.append(testkey)
    testX = np.asarray(testX)

    trainX = []
    trainY = []
    for trainkey in list(traindict.keys()):
        with open(os.path.join(datadir, 'XGBoosterfeat/taskC', type, 'train', trainkey + '.npy'), 'rb') as f:
            trainX.append(np.load(f))
            trainY.append(np.load(f))
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)

    print('data loaded')
    print('begin train')
    xg_reg = xgb.XGBClassifier(n_estimators=800, max_depth=7,
                               learning_rate=0.03, objective='multi:softmax',
                               subsample=0.9, colsample_bytree=0.85,
                               reg_lambda=10, reg_alpha=10, scale_pos_weight=15.0,
                               num_class=n_class, use_label_encoder=False)

    xg_reg.fit(trainX, trainY)
    '''xg_reg = xgb.XGBClassifier(n_estimators=800, max_depth=7,
                               learning_rate=0.03, objective='multi:softmax',
                               subsample=0.9, colsample_bytree=0.85,
                              reg_lambda=10, reg_alpha=10, scale_pos_weight=15.0,
                              num_class=3, use_label_encoder=False)
    xg_reg.load_model(os.path.join(modeldir, 'model.bin'))'''
    xg_reg.save_model(os.path.join(modeldir, 'model.bin'))
    print('end train')
    preds = xg_reg.predict(evalX)
    prob = xg_reg.predict_proba(evalX)

    #preds_onehot = one_hot(preds, n_class)
    #label_onehot = one_hot(evalY, n_class)

    '''cates = np.shape(prob)[1]
    results = []
    total_occurences = 0
    for index in range(cates):
        label = []
        predict = []
        for i in range(np.shape(prob)[0]):
            label.extend([label_onehot[i][index]])
            predict.extend([preds_onehot[i][index]])
        f1_score = compute_f1(predict, label)
        f1weight = label.count(True)
        total_occurences += f1weight
        results.append(f1_score * f1weight)
    f1 = sum(results) / total_occurences'''

    f1 = f1_score(evalY, preds, average='weighted')

    outputdict = {}
    for i in range(len(filename)):
        outputdict.update({filename[i]: {}})
        outputdict[filename[i]].update({"prob": prob[i].tolist()})
        outputdict[filename[i]].update({"predict": preds[i].tolist()})
        outputdict[filename[i]].update({"label": evalY[i].tolist()})
    with open(os.path.join(resultsdir, str(f1)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outputdict, f, ensure_ascii=False, indent=4)
    print(f1)

    preds = xg_reg.predict(testX)
    prob = xg_reg.predict_proba(testX)

    outputdict = {}
    for i in range(len(filename)):
        outputdict.update({filename[i]: {}})
        if type != 'motivation':
            outputdict[filename[i]].update({"prob": prob[i].tolist()})
            outputdict[filename[i]].update({"predict": preds[i].tolist()})
        else:
            outputdict[filename[i]].update({"prob": softmax(prob[i], axis=-1).tolist()})
            outputdict[filename[i]].update({"predict": preds[i].tolist()})
    with open(os.path.join(resultsdir, "test.json"), 'w',
              encoding='utf-8') as f:
        json.dump(outputdict, f, ensure_ascii=False, indent=4)







'''datadir = './../../../dataset'
outdir = './../../../output/taskC/train'
cashedir = './../../../CASHE'
type = 'motivation'
main(datadir, outdir, cashedir, type)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

