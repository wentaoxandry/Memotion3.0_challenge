## This script is used to read csv file and save image data in folder, save training data in json format
import os, sys
import json
import csv
import requests
import random
import numpy as np


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def readcsv(sourcedir, savedir):
    savedir = os.path.join(savedir, 'facebook')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for dset in ['test', 'train', 'dev']:
        datadict = {}
        if dset == 'train':
            jsonldir = os.path.join(sourcedir, dset + '.jsonl')
            data = []
            with open(jsonldir, 'r', encoding='utf-8') as reader:
                for line in reader:
                    data.append(json.loads(line))
        else:
            data = []
            for type in ['_seen', '_unseen']:
                jsonldir = os.path.join(sourcedir, dset + type + '.jsonl')
                with open(jsonldir, 'r', encoding='utf-8') as reader:
                    for line in reader:
                        data.append(json.loads(line))

        for i in data:
            datadict.update({i['id']: {}})
            imgsavedir = os.path.join(sourcedir, i['img'])
            datadict[i['id']].update({'imagedir': imgsavedir})
            datadict[i['id']].update({'taskA': i['label']})
            datadict[i['id']].update({'text': i['text']})

        with open(os.path.join(savedir, dset + '.json'), 'w', encoding='utf-8') as f:
            json.dump(datadict, f, ensure_ascii=False, indent=4)


'''sourcedir = './../Sourcedata/hateful_memes'
savedir = './../dataset'
readcsv(sourcedir, savedir)'''
readcsv(sys.argv[1], sys.argv[2])