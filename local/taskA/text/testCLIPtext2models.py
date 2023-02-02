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

    model = torch.load(pretraineddir, map_location=config["device"]) #CLIPtext(config["MODELtext"], config["cachedir"], 0)
    #loaded_state = torch.load(pretraineddir, map_location=config["device"]) #.replace('/Memotion3', './../../../')
    #model.load_state_dict(loaded_state)
    model.to(config['device'])


    print(len(dataset.test_dataset))

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_testclip_custom_sequence)

    print('evaluation')
    torch.cuda.empty_cache()
    model.eval()
    evalpred = []
    outpre = {}
    for i, data in enumerate(data_loader_dev, 0):
        node_sets = data[0].to(config["device"])
        mask = data[1].to(config["device"])
        filename = data[2]
        outputs = model(node_sets, mask)
        predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
        evalpred.extend(predicted.cpu().detach().tolist())


        # correct += (predicted == labels).sum().item()
        for i in range(len(filename)):
            outpre.update({filename[i]: {}})
            outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
            outpre[filename[i]].update(
                {'prob': torch.softmax(outputs[i], dim=-1).cpu().detach().data.numpy().tolist()})

    with open(os.path.join(config["savedir"], "test" + '_' + config["i"] +".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)







def maxlist(list1,list2): #def returns 3rd list
    list3 = [max(value) for value in zip(list1, list2)]
    return list3

def main(datadir, savedir, ensembledir, cashedir, bestmodelrayinfo):
    res = tuple(bestmodelrayinfo.split(', '))
    pretraindirs = [res[5].strip(')').strip('\'').replace('./', '/Memotion3/'),
                    res[11].strip(')').strip('\'').replace('./', '/Memotion3/').strip(')]\'')]
    bs = [int(res[4]), int(res[10])]
    max_num_epochs = 70
    best_model = 'acc'
    N_AVERAGE = 1
    ## 4-folder cross-validation
    cvensembledir = os.path.join(ensembledir, 'text')
    if not os.path.exists(cvensembledir):
        os.makedirs(cvensembledir)
    modeldir = os.path.join(savedir, 'text', 'model')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    ## load dev features
    with open(os.path.join(datadir, "memotion3", "test_en.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    tokenizer = CLIPTokenizer.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = BERTweetevaldatasetclass(test_file=testdict,
                                       dset='test',
                                       tokenizer=tokenizer,
                                       device=device,
                                       max_len=max_len)
    for i in range(len(pretraindirs)):
        config = {
            "MODELtext": MODELtext,
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "lr": 1e-5,
            "modeldir": modeldir,
            "pretraineddir": pretraindirs[i],
            "savedir": cvensembledir,
            "batch_size": bs[i],  # 16,
            "cachedir": cashedir,
            "epochs": max_num_epochs,
            "best_model": best_model,
            "N_AVERAGE": N_AVERAGE,
            "i": str(i)
        }
        eval(config, dataset)






'''datadir = './../../../dataset'
outdir = './../../../output/taskA/train'
cashedir = './../../../CASHE'
ensembledir = './../../../Ensemble/taskA'
bestmodelrayinfo = "(0.5180849528702725, 2e-05, 0, 0.01, 32, './RAY_results/taskA_train_text_RAY/training_2022-11-14_16-17-02/training_c64b1_00004_4_lr=2e-05,weight_decay=0.01_2022-11-14_16-19-24/checkpoint_000003/checkpoint')"
main(datadir, outdir, ensembledir, cashedir, bestmodelrayinfo)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


