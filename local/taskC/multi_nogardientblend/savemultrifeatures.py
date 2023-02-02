import os, sys, json
import random
from modelxgboost import CLIPmulti
from utilsxgboost import *
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


def eval(config, dataset=None):
    modeldir = config["modeldir"]
    pretraineddir = config["pretraineddir"]
    model = CLIPmulti(config["MODELtext"], config["cachedir"], config["dropout"], type=config["type"])
    loaded_state = torch.load(pretraineddir, map_location=config['device']).state_dict()
    model.load_state_dict(loaded_state)
    # model = torch.load(pretraineddir, map_location=config['device'])
    '''if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)'''
    model.to(config['device'])

    print(len(dataset.test_dataset))

    if config["dset"] == 'test':
        data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_testclip_custom_sequence)

        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        model.eval()
        outpre = {}
        total = 0
        evalpredict = []
        evallabel = []
        for i, data in enumerate(data_loader_dev, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            filename = data[3]
            _, emb = model(node_sets, mask, image, 1000)

            for i in range(len(filename)):
                with open(os.path.join(config["savedir"], filename[i] + '.npy'), 'wb') as f:
                    np.save(f, emb[i].cpu().detach().data.numpy())

    else:
        data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_clip_custom_sequence)


        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        model.eval()
        outpre = {}
        total = 0
        evalpredict = []
        evallabel = []
        for i, data in enumerate(data_loader_dev, 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            label = data[3].to(config["device"])
            filename = data[4]
            labels = label.squeeze(-1)
            _, emb = model(node_sets, mask, image, 1000)

            for i in range(len(filename)):
                with open(os.path.join(config["savedir"], filename[i] + '.npy'), 'wb') as f:
                    np.save(f, emb[i].cpu().detach().data.numpy())
                    np.save(f, labels[i].cpu().detach().data.numpy())


def main(datadir, outdir, cashedir, type, bestmodelrayinfo):
    res = tuple(bestmodelrayinfo.split(', '))
    pretraineddir = res[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    bs = int(res[4])
    npdatadir = os.path.join(datadir, 'pretrainCLIP')


    ## 4-folder cross-validation
    savedir = os.path.join(datadir, 'XGBoosterfeat', 'taskC', type)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    modeldir = os.path.join(outdir, 'multi', type, 'model')

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "memotion3", "train_en.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "memotion3", "val_en.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)
    with open(os.path.join(datadir, "memotion3", "test_en.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"

    for dset in ['train', 'val', 'test']:
        savedatadir = os.path.join(savedir, dset)
        if not os.path.exists(savedatadir):
            os.makedirs(savedatadir)

        if dset == 'train':
            datadict = traindict
        elif dset == 'val':
            datadict = valdict
        elif dset == 'test':
            datadict = testdict

        dataset = BERTweetevalsetclass(test_file=datadict,
                                       dset=dset,
                                       device=device,
                                       npdatadir=npdatadir,
                                       type=type)

        config = {
            "MODELtext": MODELtext,
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "dropout": 0,
            "modeldir": modeldir,
            "pretraineddir": pretraineddir,
            "savedir": savedatadir,
            "batch_size": bs,  # 16,
            "cachedir": cashedir,
            "dset": dset,
            "type": type
        }
        eval(config, dataset)











'''datadir = './../../../dataset'
outdir = './../../../output/taskC/train'
cashedir = './../../../CASHE'
type = 'motivation'
main(datadir, outdir, cashedir, type)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


