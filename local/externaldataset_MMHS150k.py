import os, sys
import json
import random

def get_memes(sourcedir, savedir):
    savedir = os.path.join(savedir, 'MMHS150k')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    memesinfodir = os.path.join(sourcedir, 'img_txt')
    memesnames = os.listdir(memesinfodir)
    imgdir = os.path.join(sourcedir, 'img_resized')
    dset = {}
    numimg = 0
    for memename in memesnames:
        datadir = os.path.join(memesinfodir, memename)
        with open(datadir, encoding="utf8") as json_file:
            data = json.load(json_file)
        memename = memename.split('.')[0]
        dset.update({memename: {}})
        dset[memename].update({'text': ' '.join(data['img_text']).lower()})
        dset[memename].update({'imagedir': os.path.join(imgdir, memename + '.jpg')})
        print(str(numimg) + '/' + str(len(memesnames)))

        numimg = numimg + 1

    with open(os.path.join(savedir, 'MMHS150k.json'), 'w', encoding='utf-8') as f:
        json.dump(dset, f, ensure_ascii=False, indent=4)



'''sourcedir = './../external_dataset/MMHS150k'
savedir = './../dataset'
get_memes(sourcedir, savedir)'''
get_memes(sys.argv[1], sys.argv[2])

