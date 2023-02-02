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

class GradientBlending(torch.nn.Module):
    def __init__(self, text_weight=0.0, image_weight=0.0, multi_weight=1.0, loss_scale=1.0):
        "Expects weights for each model, the combined model, and an overall scale"
        super(GradientBlending, self).__init__()
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.multi_weight = multi_weight
        self.scale = loss_scale
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, text_out, image_out, multi_out, targ):
        "Gathers `self.loss` for each model, weighs, then sums"
        multi_loss = self.ce(multi_out, targ) * self.scale
        text_loss = self.ce(text_out, targ) * self.scale
        image_loss = self.ce(image_out, targ) * self.scale
        weighted_text_loss = text_loss * self.text_weight
        weighted_image_loss = image_loss * self.image_weight
        weighted_multi_loss = multi_loss * self.multi_weight
        loss = weighted_text_loss + weighted_image_loss + weighted_multi_loss
        return loss

def training(config, dataset=None):
    iw = config["img_weight"]
    mw = config["multi_weight"]

    uniweight = 1 - mw
    if uniweight > iw:
        tw = 1 - mw - iw
    else:
        iw = uniweight
        tw = 0

    criterion = GradientBlending(text_weight=tw, image_weight=iw, multi_weight=mw, loss_scale=1.0)
    model = CLIPmulti(config["MODELtext"], config["cachedir"], config["dropout"], config["raychecktextdir"], config["raycheckimagedir"], type=config["type"])

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
        '''if epoch < 5:
            for param in model.clipimage.parameters():
                param.requires_grad = False
            for param in model.cliptext.parameters():
                param.requires_grad = False
        else:
            for param in model.clipimage.parameters():
                param.requires_grad = True
            for param in model.cliptext.parameters():
                param.requires_grad = True'''
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
            textlogit, imagelogit, multilogit = model(node_sets, mask, image, epoch)
            label = label.squeeze(-1)
            loss = criterion(textlogit, imagelogit, multilogit, label)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            #print("\r%f" % loss, end='')
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
        predictlist = []
        model.eval()
        for i, data in enumerate(data_loader_dev, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            label = data[3].to(config["device"])
            labels = label.squeeze(-1)
            textlogit, imagelogit, multilogit = model(node_sets, mask, image, epoch)

            predicted = torch.argmax(torch.softmax(multilogit, dim=-1), dim=-1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            labellist.extend(labels.cpu().data.numpy().tolist())
            predictlist.extend(predicted.cpu().detach().data.numpy().tolist())

            loss = criterion(textlogit, imagelogit, multilogit, labels)
            val_loss += loss.cpu().detach().numpy()
            val_steps += 1
            #del loss, mask
        trainallscore = f1_score(labellist, predictlist, average='weighted')
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(model, path)

        tune.report(loss=(val_loss / val_steps), accuracy=trainallscore)
        print("Finished Training")


def RAY_find(Datadir, cashedir, type, besttextmodelinfo, bestimagemodelinfo):
    textres = tuple(besttextmodelinfo.split(', '))
    raychecktextdir = textres[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    imageres = tuple(bestimagemodelinfo.split(', '))
    raycheckimagedir = imageres[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    npdatadir = os.path.join(Datadir.replace('./', '/Memotion3/'), 'pretrainCLIP')
    max_num_epochs = 15


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

    dataset = BERTweetdatasetclass(train_file=traindict,
                                   test_file=valdict,
                                   device=device,
                                   npdatadir=npdatadir,
                                   type=type)

    ## Configuration
    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "eps": 1e-8,
        "cachedir": cashedir,
        "lr": 2e-6,
        "dropout": 0,
        "weight_decay": tune.choice([0, 1e-3, 0.01]),
        "batch_size": 32,
        "img_weight": tune.choice([0.6, 0.5, 0.3]),
        "multi_weight": tune.choice([0.3, 0.4, 0.5]),
        "epochs": max_num_epochs,
        "raychecktextdir": raychecktextdir,
        "raycheckimagedir": raycheckimagedir,
        "type": type
    }

    '''config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "weight_decay": 0,
        "eps": 1e-8,
        "cachedir": cashedir,
        "lr": 1e-5,
        "batch_size": 32,
        "img_weight": 0.3,
        "multi_weight": 0.5,
        "dropout": 0,
        "epochs": 3,
        "raychecktextdir": raychecktextdir,
        "raycheckimagedir": raycheckimagedir,
        "type": type,
        "gamma": 1
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
#outdir = './../output/text/taskA/pretrain'
cashedir = './../../../CASHE'
type = 'motivation'
RAY_find(datadir, cashedir, type)'''
RAY_find(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


