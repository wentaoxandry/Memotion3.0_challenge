import os, sys, json
import random
from model import *
from utils import *
import numpy as np
import collections
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
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

def eval(config, dataset=None):
    modeldir = config["modeldir"]
    pretraineddir = config["pretraineddir"]
    model = torch.load(pretraineddir, map_location=config['device'])

    # if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = torch.nn.DataParallel(model)
    model.to(config['device'])

    print(len(dataset.test_dataset))

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_clip_custom_sequence)

    # Validation loss
    print('evaluation')
    model.eval()
    evalpred = []
    evallabel = []
    outpre = {}
    total = 0
    for i, data in enumerate(data_loader_dev, 0):
        image = data[0].to(config["device"])
        label = data[1].to(config["device"])
        filename = data[2]
        labels = label.squeeze(-1)
        outputs = model(image)
        predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
        evalpred.extend(predicted.cpu().detach().tolist())
        evallabel.extend(labels.cpu().detach().data.numpy().tolist())
        total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        for i in range(len(filename)):
            outpre.update({filename[i]: {}})
            outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
            outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
            outpre[filename[i]].update(
                {'prob': torch.softmax(outputs[i], dim=-1).cpu().detach().data.numpy().tolist()})

        # del dev_loss, outputs, node_sets, mask, labels

    allscore = f1_score(evallabel, evalpred, average='weighted')
    OUTPUT_DIR = os.path.join(modeldir, str(
                                  allscore)[:6] + '.pkl')
    torch.save(model, OUTPUT_DIR)
    with open(os.path.join(config["savedir"], str(allscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)





def main(datadir, savedir, ensembledir, cashedir, bestmodelrayinfo):
    res = tuple(bestmodelrayinfo.split(', '))
    pretraineddir = res[-1].strip(')').strip('\'').replace('./', '/Memotion3/')
    bs = int(res[4])
    npdatadir = os.path.join(datadir.replace('./', '/Memotion3/'), 'pretrainCLIP') #
    N_AVERAGE = 1

    ## 4-folder cross-validation
    cvensembledir = os.path.join(ensembledir, 'image')
    if not os.path.exists(cvensembledir):
        os.makedirs(cvensembledir)
    modeldir = os.path.join(savedir, 'image', 'model')
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "memotion3", "val_en.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    dataset = BERTweetmemotionevalclass(test_file=valdict,
                                        dset='val',
                                        device=device,
                                        max_len=max_len,
                                        npdatadir=npdatadir)

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
        "N_AVERAGE": N_AVERAGE
    }
    eval(config, dataset)










'''datadir = './../../../dataset'
outdir = './../../../output/taskA/train'
cashedir = './../../../CASHE'
ensembledir = './../../../Ensemble/taskA'
bestmodelrayinfo = "(0.5097444140956332, 1e-05, 0.5, 0, 16, './RAY_results/taskA_train_image_RAY/training_2022-11-15_08-09-08/training_c7bb4_00010_10_batch_size=16,dropout=0.5,lr=1e-05,weight_decay=0_2022-11-15_08-25-39/checkpoint_000003/checkpoint')"
main(datadir, outdir, ensembledir, cashedir, bestmodelrayinfo)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


