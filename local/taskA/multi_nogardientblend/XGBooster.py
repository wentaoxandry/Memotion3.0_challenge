import os, sys, json
import random
import collections
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
#import csv
SEED=666
random.seed(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix
def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1


def main(datadir, savedir, cashedir):
    ## 4-folder cross-validation
    for cv in range(4):

        modeldir = os.path.join(savedir, 'XGBooster', str(cv), 'model')
        resultsdir = os.path.join(savedir, 'XGBooster', str(cv), 'results')

        for makedir in [modeldir, resultsdir, cashedir]:
            if not os.path.exists(makedir):
                os.makedirs(makedir)


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
        testX = []
        testY = []
        filename = []
        for testkey in list(newtestdict.keys()):
            with open(os.path.join(datadir, 'XGBoosterfeat/taskA', str(cv), testkey + '.npy'), 'rb') as f:
                testX.append(np.load(f))
                testY.append(np.load(f))
                filename.append(testkey)
        testX = np.asarray(testX)
        testY = np.asarray(testY)



        trainX = []
        trainY = []
        for trainkey in list(newtraindict.keys()):
            with open(os.path.join(datadir, 'XGBoosterfeat/taskA', str(cv), trainkey + '.npy'), 'rb') as f:
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
                                  num_class=3, use_label_encoder=False)

        xg_reg.fit(trainX, trainY)
        '''xg_reg = xgb.XGBClassifier(n_estimators=800, max_depth=7,
                                   learning_rate=0.03, objective='multi:softmax',
                                   subsample=0.9, colsample_bytree=0.85,
                                  reg_lambda=10, reg_alpha=10, scale_pos_weight=15.0,
                                  num_class=3, use_label_encoder=False)
        xg_reg.load_model(os.path.join(modeldir, 'model.bin'))'''
        xg_reg.save_model(os.path.join(modeldir, 'model.bin'))
        print('end train')
        preds = xg_reg.predict(testX)
        prob = xg_reg.predict_proba(testX)
        f1 = compute_f1(preds.tolist(), testY.tolist())
        outputdict = {}
        for i in range(len(filename)):
            outputdict.update({filename[i]: {}})
            outputdict[filename[i]].update({"prob": prob[i].tolist()})
            outputdict[filename[i]].update({"predict": preds[i].tolist()})
            outputdict[filename[i]].update({"label": testY[i].tolist()})
        with open(os.path.join(resultsdir, str(f1)[:6] + ".json"), 'w',
                  encoding='utf-8') as f:
            json.dump(outputdict, f, ensure_ascii=False, indent=4)
        print(f1)


'''datadir = './../../../dataset'
outdir = './../../../output/taskA/train'
cashedir = './../../../CASHE'
main(datadir, outdir, cashedir)'''
main(sys.argv[1], sys.argv[2], sys.argv[3])


