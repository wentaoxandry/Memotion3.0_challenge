import os, sys, json
import random
from model import *
from utils import *
from transformers import get_linear_schedule_with_warmup, CLIPProcessor
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
    model = CLIP(config["MODELtext"], config["cachedir"])

    loaded_state = torch.load(config["pretraineddir"], map_location=config["device"])[0]
    #self_state = model.state_dict()

    #loaded_state = {k: v for k, v in loaded_state.items() if not k.startswith('output')}
    #self_state.update(loaded_state)
    model.load_state_dict(loaded_state)

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
                                                    train_examples_len / config["batch_size"]) * int(0.2 * config["epochs"]),
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
        running_loss = 0.0
        epoch_steps = 0
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
            #print("\r%f" % loss, end='')
            # print statistics
            running_loss += loss.mean().cpu().detach().numpy()
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
        for i, data in enumerate(data_loader_dev, 0):
            with torch.no_grad():
                node_sets = data[0].to(config["device"])
                mask = data[1].to(config["device"])
                pixel = data[2].to(config["device"])
                loss = model(node_sets, mask, pixel)

                val_loss += loss.mean().cpu().detach().numpy()
                val_steps += 1
                del loss, mask

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
        print("Finished Training")


def RAY_find(Datadir, cashedir, bestmodelinfo):
    res = tuple(bestmodelinfo.split(', '))
    pretraineddir = res[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    max_num_epochs = 15
    npdatadir = os.path.join(Datadir, 'pretrainCLIP')
    #max_num_epochs = 10

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    with open(os.path.join(Datadir, "pretrain_CLIP_train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)

    ## load dev features
    with open(os.path.join(Datadir, "pretrain_CLIP_test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    dataset = CLIPdatasetclass(train_file=traindict,
                                   test_file=testdict,
                                   device=device,
                                   max_len=max_len,
                               npdatadir=npdatadir)

    ## Configuration
    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "eps": 1e-8,
        "cachedir": cashedir,
        "lr": tune.choice([2e-6, 1e-6]),
        "weight_decay": 0,
        "batch_size": tune.choice([32, 64]),
        "epochs": max_num_epochs,
        "pretraineddir": pretraineddir
    }

    '''config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "weight_decay": 0,
        "eps": 1e-8,
        "cachedir": cashedir,
        "lr": 1e-5,
        "batch_size": 2,
        "epochs": 3
    }
    training(config, dataset=dataset)'''

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    #training(config, dataset=dataset)
    result = tune.run(
        tune.with_parameters(training, dataset=dataset),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=3,
        local_dir="./RAY",
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))






'''datadir = './../../dataset'
#outdir = './../output/text/taskA/pretrain'
cashedir = './../../CASHE'
RAY_find(datadir, cashedir)'''
RAY_find(sys.argv[1], sys.argv[2], sys.argv[3])


