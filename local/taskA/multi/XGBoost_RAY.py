import os, sys, json
import random
import xgboost as xgb
from utils import *
from torch.utils.data import DataLoader, Dataset
from ray import tune, ray_constants
from ray.tune import CLIReporter
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
from ray.tune.schedulers import ASHAScheduler
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
    trainX = dataset['trainX']
    trainY = dataset['trainY']
    testX = dataset['testX']
    testY = dataset['testY']
    xg_reg = xgb.XGBClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"],
                               learning_rate=config["learning_rate"], objective='multi:softmax',
                               subsample=config["subsample"], colsample_bytree=config["colsample_bytree"],
                               reg_lambda=config["reg_lambda"], reg_alpha=config["reg_alpha"], scale_pos_weight=config["scale_pos_weight"],
                               num_class=3, use_label_encoder=False)

    xg_reg.fit(trainX, trainY)
    preds = xg_reg.predict(testX)
    prob = xg_reg.predict_proba(testX)
    f1 = compute_f1(preds.tolist(), testY.tolist())
    with tune.checkpoint_dir(0) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        xg_reg.save_model(path)

    tune.report(accuracy=f1)
    print("Finished Training")


def RAY_find(Datadir, cashedir):

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    with open(os.path.join(Datadir, "memotion3", "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(Datadir, "memotion3", 'cvtest0.txt')) as f:
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
    dataset = {}
    dataset.update({'train': newtraindict})
    dataset.update({'test': newtestdict})
    datadict = {}
    testX = []
    testY = []
    filename = []
    for testkey in list(dataset['test'].keys()):
        with open(os.path.join(Datadir, 'XGBoosterfeat/taskA', str(0), testkey + '.npy'), 'rb') as f:
            testX.append(np.load(f))
            testY.append(np.load(f))
            filename.append(testkey)
    testX = np.asarray(testX)
    testY = np.asarray(testY)

    trainX = []
    trainY = []
    for trainkey in list(dataset['train'].keys()):
        with open(os.path.join(Datadir, 'XGBoosterfeat/taskA', str(0), trainkey + '.npy'), 'rb') as f:
            trainX.append(np.load(f))
            trainY.append(np.load(f))
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)

    print('data loaded')
    datadict.update({'trainX': trainX})
    datadict.update({'trainY': trainY})
    datadict.update({'testX': testX})
    datadict.update({'testY': testY})



    ## Configuration
    config = {
        "n_estimators": tune.choice([2e-3, 2e-4, 2e-5]),
        "max_depth": tune.choice([2e-3, 2e-4, 2e-5]),
        "learning_rate": tune.choice([2e-3, 2e-4, 2e-5]),
        "subsample": tune.choice([2e-3, 2e-4, 2e-5]),
        "colsample_bytree": tune.choice([2e-3, 2e-4, 2e-5]),
        "reg_lambda": tune.choice([2e-3, 2e-4, 2e-5]),
        "reg_alpha": tune.choice([2e-3, 2e-4, 2e-5]),
        "scale_pos_weight": tune.choice([2e-3, 2e-4, 2e-5])
    }
    '''config = {
        "n_estimators": 1,
        "max_depth": 1,
        "learning_rate": 0.1,
        "subsample": 0.2,
        "colsample_bytree": 0.85,
        "reg_lambda": 1,
        "reg_alpha": 1,
        "scale_pos_weight": 1.0
    }
    training(config, dataset=datadict)'''

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=100,
        grace_period=1,
        reduction_factor=2)
    #training(config, dataset=dataset)
    result = tune.run(
        tune.with_parameters(training, dataset=datadict),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=10,
        local_dir="./RAY",
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))






'''datadir = './../../../dataset'
#outdir = './../output/text/taskA/train'
cashedir = './../../../CASHE'
RAY_find(datadir, cashedir)'''
RAY_find(sys.argv[1], sys.argv[2])


