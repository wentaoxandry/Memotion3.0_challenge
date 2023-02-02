import os, sys, json
import random
from model import *
from utils import *
import numpy as np
import collections
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
    model = torch.load(pretraineddir, map_location=config["device"])

    model.to(config['device'])
    print(len(dataset.test_dataset))

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_testclip_custom_sequence)

    # Validation loss
    print('evaluation')
    torch.cuda.empty_cache()
    model.eval()
    correct = 0
    outpre = {}
    evalpredict = []
    for i, data in enumerate(data_loader_dev, 0):
        node_sets = data[0]
        mask = data[1]
        filename = data[2]
        outputs = model(node_sets, mask)

        predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
        for i in range(len(filename)):
            outpre.update({filename[i]: {}})
            outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
            outpre[filename[i]].update({'prob': torch.softmax(outputs[i], dim=-1).cpu().detach().tolist()})
        evalpredict.extend(one_hot(predicted.cpu().detach().data.numpy(), outputs.size(1)).tolist())
    # del dev_loss, outputs, node_sets, mask, labels

    with open(os.path.join(config["savedir"], "test" + '_' + config["i"] +".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)



def main(datadir, savedir, ensembledir, cashedir, type, bestmodelrayinfo):
    res = tuple(bestmodelrayinfo.split(', '))
    pretraindirs = [res[6].strip(')').strip('\'').replace('./', '/Memotion3/'),
                    res[13].strip(')').strip('\'').replace('./', '/Memotion3/').strip(')]\'')]
    bs = [int(res[4]), int(res[11])]
    ## 4-folder cross-validation
    cvensembledir = os.path.join(ensembledir, 'text', type)
    if not os.path.exists(cvensembledir):
        os.makedirs(cvensembledir)

    modeldir = os.path.join(savedir, 'text', type, 'model')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    print(modeldir)
    resultsdir = os.path.join(savedir, 'text')

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "memotion3", "test_en.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    tokenizer = CLIPTokenizer.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = BERTweetevalsetclass(test_file=testdict,
                                   dset='test',
                                   tokenizer=tokenizer,
                                   device=device,
                                   max_len=max_len,
                                   type=type)
    for i in range(len(pretraindirs)):
        config = {
            "MODELtext": MODELtext,
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "modeldir": modeldir,
            "pretraineddir": pretraindirs[i],
            "resultsdir": resultsdir,
            "savedir": cvensembledir,
            "batch_size": bs[i],  # 16,
            "cachedir": cashedir,
            "type": type,
            "i": str(i)
        }
        eval(config, dataset)











'''datadir = './../../../dataset'
outdir = './../../../output/taskC/train'
cashedir = './../../../CASHE'
type = 'motivation'
ensembledir = './../../../Ensemble/taskC'
main(datadir, outdir, ensembledir, cashedir, type)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


