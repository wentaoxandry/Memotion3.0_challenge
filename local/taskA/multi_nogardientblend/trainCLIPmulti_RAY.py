import os, sys, json
import random
from model import *
from utils import *
from transformers import get_linear_schedule_with_warmup, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset
from ray import tune, ray_constants
from ray.tune import CLIReporter
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import f1_score
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
    model = CLIPmulti(config["MODELtext"], config["cachedir"], config["dropout"], config["raychecktextdir"], config["raycheckimagedir"])

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
                                                    train_examples_len / config["batch_size"]) * int(0.2 * config["epochs"]),
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])'''
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
        running_loss = 0.0
        epoch_steps = 0
        model.train()

        for i, data in enumerate(data_loader_train, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            label = data[3].to(config["device"])
            filename = data[4]
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask, image, epoch)
            label = label.squeeze(-1)
            loss = criterion(outputs, label)
            #print("\r%f" % loss, end='')
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # print statistics
            running_loss += loss.cpu().detach().numpy()
            epoch_steps += 1
            if i % 200 == 199:  # print every 200 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
                epoch_steps = 0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        labellist = []
        model.eval()
        predictlist = []
        for i, data in enumerate(data_loader_dev, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            label = data[3].to(config["device"])
            labels = label.squeeze(-1)
            outputs = model(node_sets, mask, image, epoch)

            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            labellist.extend(labels.cpu().data.numpy().tolist())
            predictlist.extend(predicted.cpu().detach().tolist())



            loss = criterion(outputs, labels)
            val_loss += loss.cpu().detach().numpy()
            val_steps += 1
            #del loss, mask
        
        trainallscore = f1_score(labellist, predictlist, average='weighted')
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(model, path)

        tune.report(loss=(val_loss / val_steps), accuracy=trainallscore)
        print("Finished Training")


def RAY_find(Datadir, cashedir, besttextmodelinfo, bestimagemodelinfo):
    npdatadir = os.path.join(Datadir.replace('./', '/Memotion3/'), 'pretrainCLIP') #.replace('./', '/Memotion3/')
    max_num_epochs = 15

    textres = tuple(besttextmodelinfo.split(', '))
    raychecktextdir = textres[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    imageres = tuple(bestimagemodelinfo.split(', '))
    raycheckimagedir = imageres[-1].strip(')').strip('\'').replace('./', '/Memotion3/')


    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    with open(os.path.join(Datadir, "memotion3", "train_en.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(Datadir, "memotion3", "val_en.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    dataset = BERTweetmemotionclass(train_file=traindict,
                                   test_file=valdict,
                                   device=device,
                                   max_len=max_len,
                                   npdatadir=npdatadir)

    ###Configuration
    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "eps": 1e-8,
        "cachedir": cashedir,
        "lr": tune.choice([2e-5, 1e-5, 2e-6]),
        "dropout": 0,
        "weight_decay": tune.choice([0, 1e-3, 0.01]),
        "batch_size": 32,
        "epochs": max_num_epochs,
        "raychecktextdir": raychecktextdir,
        "raycheckimagedir": raycheckimagedir
    }

    '''config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "weight_decay": 0,
        "eps": 1e-8,
        "cachedir": cashedir,
        "lr": 1e-5,
        "batch_size": 10,
        "dropout": 0,
        "epochs": 3,
        "raychecktextdir": raychecktextdir,
        "raycheckimagedir": raycheckimagedir
    }
    training(config, dataset=dataset)'''

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        grace_period=10,
        reduction_factor=3,
        brackets=1)
    # training(config, dataset=dataset)
    result = tune.run(
        tune.with_parameters(training, dataset=dataset),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=9,
        local_dir="./RAY",
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))






'''datadir = './../../../dataset'
cashedir = './../../../CASHE'
besttextmodelinfo = "(0.5542730627306274, 1e-05, 0, 0.001, 32, './RAY_results/taskA_train_text_RAY/training_2022-11-15_14-08-21/training_f65c5_00005_5_lr=1e-05,weight_decay=0.001_2022-11-15_14-16-41/checkpoint_000014/checkpoint')"
bestimagemodelinfo = "(0.5382120605371088, 1e-05, 0, 0.01, 8, './RAY_results/taskA_train_image_RAY/training_2022-11-15_18-22-01/training_660e0_00004_4_batch_size=8,lr=1e-05,weight_decay=0.01_2022-11-15_18-37-02/checkpoint_000006/checkpoint')"
RAY_find(datadir, cashedir, besttextmodelinfo, bestimagemodelinfo)'''
RAY_find(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


