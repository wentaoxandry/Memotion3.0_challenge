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


def eval(config, dataset=None):
    modeldir = config["modeldir"]
    pretraineddir = config["pretraineddir"]
    model = torch.load(pretraineddir, map_location=config['device'])

    '''if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)'''
    model.to(config['device'])

    print(len(dataset.test_dataset))

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_testclip_custom_sequence)

    # Validation loss
    print('evaluation')
    model.eval()
    outpre = {}
    for i, data in enumerate(data_loader_dev, 0):
        node_sets = data[0].to(config["device"])
        mask = data[1].to(config["device"])
        image = data[2].to(config["device"])
        filename = data[3]
        textlogit, imagelogit, multilogit = model(node_sets, mask, image, 10000)

        predicted = torch.argmax(torch.softmax(multilogit, dim=-1), dim=-1)
        for i in range(len(filename)):
            outpre.update({filename[i]: {}})
            outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
            outpre[filename[i]].update({'prob': torch.softmax(multilogit[i], dim=-1).cpu().detach().tolist()})
    # del dev_loss, outputs, node_sets, mask, labels

    with open(os.path.join(config["savedir"], "test.json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)




def main(datadir, savedir, ensembledir, cashedir, type, bestmodelrayinfo):
    res = tuple(bestmodelrayinfo.split(', '))
    pretraineddir = res[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    bs = int(res[4])
    npdatadir = os.path.join(datadir, 'pretrainCLIP')
    N_AVERAGE = 1


    ## 4-folder cross-validation
    cvensembledir = os.path.join(ensembledir, 'multi', type)
    if not os.path.exists(cvensembledir):
        os.makedirs(cvensembledir)
    modeldir = os.path.join(savedir, 'multi', type, 'model')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "memotion3", "test_en.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"

    dataset = BERTweetevalsetclass(test_file=testdict,
                                   dset='test',
                                   device=device,
                                   npdatadir=npdatadir,
                                   type=type)

    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "weight_decay": 0,
        "eps": 1e-8,
        "modeldir": modeldir,
        "pretraineddir": pretraineddir,
        "savedir": cvensembledir,
        "batch_size": bs,  # 16,
        "cachedir": cashedir,
        "type": type
    }
    eval(config, dataset)









'''datadir = './../../../dataset'
outdir = './../../../output/taskC/train'
cashedir = './../../../CASHE'
type = 'motivation'
ensembledir = './../../../Ensemble/taskC'
main(datadir, outdir, ensembledir, cashedir, type)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


