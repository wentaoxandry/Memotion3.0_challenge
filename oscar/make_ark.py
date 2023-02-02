import json
import pickle, os
import sys
import gzip
import torch
import json
import numpy as np
from kaldiio import WriteHelper
import argparse




def make_ark():
    parser = argparse.ArgumentParser(description='Creak ark files')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='directory to load data',
                        default="output")
    parser.add_argument('--output_dir', dest='output_dir',
                        help='directory to load data',
                        default="Dataset")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    boxdir = os.path.join(args.data_dir, 'boxes')
    featdir = os.path.join(args.data_dir, 'features')
    jsondir = os.path.join(args.data_dir, 'json')

    filelist = os.listdir(boxdir)
    #filelist = filelist[:100]

    boxarksavedir = 'ark,scp:' + os.path.join(args.output_dir,
                                           'boxes.ark') + ',' + os.path.join(
        args.output_dir, 'boxes.scp')

    featarksavedir = 'ark,scp:' + os.path.join(args.output_dir,
                                           'feats.ark') + ',' + os.path.join(
        args.output_dir, 'feats.scp')

    with WriteHelper(boxarksavedir, compression_method=2) as writer:

        for datafile in filelist:
            filename = datafile.split('.')[0]
            with open(os.path.join(boxdir, datafile), 'rb') as f:
                boxdata = pickle.load(f)

            writer(filename, boxdata)

    with WriteHelper(featarksavedir, compression_method=2) as writer:

        for datafile in filelist:
            filename = datafile.split('.')[0]
            with open(os.path.join(featdir, datafile), 'rb') as f:
                featdata = pickle.load(f)

            writer(filename, featdata)

    outdict = {}
    for datafile in filelist:
        filename = datafile.split('.')[0]
        with open(os.path.join(jsondir, filename + ".json"), encoding="utf8") as json_file:
            jsondata = json.load(json_file)
        outdict.update(jsondata)

    boxscpdir = os.path.join(args.output_dir, 'boxes.scp')
    featscpdir = os.path.join(args.output_dir, 'feats.scp')
    with open(boxscpdir) as f:
        boxsrcdata = f.readlines()
    with open(featscpdir) as f:
        featsrcdata = f.readlines()
    boxsrcdict = {}
    for j in boxsrcdata:
        boxsrcdict.update({j.split(' ')[0]: j.split(' ')[1]})
    featsrcdict = {}
    for j in featsrcdata:
        featsrcdict.update({j.split(' ')[0]: j.split(' ')[1]})

    dsetkeys = outdict.keys()
    for dsetkey in dsetkeys:
        outdict[dsetkey]['features'] = featsrcdict[dsetkey]
        outdict[dsetkey]['boxes'] = boxsrcdict[dsetkey]

    with open(os.path.join(args.output_dir, 'data.json'), 'w', encoding='utf-8') as f:
        json.dump(outdict, f, ensure_ascii=False, indent=4)





if __name__ == "__main__":
    make_ark()
