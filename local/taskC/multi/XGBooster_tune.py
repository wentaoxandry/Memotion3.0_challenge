import os, sys, json
import random
from model import *
from utils import *
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
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

    max_depth = 20
    reg_lambda = 10
    reg_alpha = 10
    num_class = 3
    objective = 'multi:softmax'
    use_label_encoder = False
    subsample = 0.9
    colsample_bytree = 0.85
    model = xgb.XGBClassifier(max_depth=max_depth,
                               objective=objective,
                               subsample=subsample, colsample_bytree=colsample_bytree,
                               reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                               num_class=num_class, use_label_encoder=use_label_encoder)

    learning_rate = np.arange(3e-4, 0.3, 0.1)
    scale_pos_weight = np.arange(1.0, 17.0, 5.0)
    n_estimators = [800, 900, 1000]

    param_grid = dict(learning_rate=learning_rate,
                      scale_pos_weight=scale_pos_weight,
                      n_estimators=n_estimators)

    grid_search = GridSearchCV(model, param_grid, scoring="f1_weighted", n_jobs=16)
    grid_result = grid_search.fit(trainX, trainY)
    best_param = grid_result.best_params_
    print(best_param)
    xg_reg = xgb.XGBClassifier(best_param)

    '''xg_reg = xgb.XGBClassifier(n_estimators=800, max_depth=7,
                               learning_rate=0.03, objective='multi:softmax',
                               subsample=0.9, colsample_bytree=0.85,
                              reg_lambda=10, reg_alpha=10, scale_pos_weight=15.0,
                              num_class=3, use_label_encoder=False)'''

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

    preds_onehot = one_hot(preds, n_class)
    label_onehot = one_hot(evalY, n_class)

    cates = np.shape(prob)[1]
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
    f1 = sum(results) / total_occurences

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
        outputdict[filename[i]].update({"prob": prob[i].tolist()})
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

