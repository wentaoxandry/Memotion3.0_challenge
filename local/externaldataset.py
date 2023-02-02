import os, sys
import json
import random

def get_memes(sourcedir, savedir):
    savedir = os.path.join(savedir, 'external')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    memesinfodir = os.path.join(sourcedir, 'dataset', 'memes')
    memesnames = os.listdir(memesinfodir)
    imgdir = os.path.join(sourcedir, 'dataset', 'templates', 'img')
    dset = {}
    numimg = 0
    for memename in memesnames:
        datadir = os.path.join(memesinfodir, memename)
        with open(datadir, encoding="utf8") as json_file:
            data = json.load(json_file)
        memename = memename.split('.')[0]
        nummeme = 0
        for i in range(len(data)):
            dset.update({memename + '_' + str(i): {}})
            dset[memename + '_' + str(i)].update({'text': ' '.join(data[i]['boxes']).lower()})
            dset[memename + '_' + str(i)].update({'imagedir': os.path.join(imgdir, memename + '.jpg')})
            print(str(nummeme) + '/' + str(len(data)) + ', ' + str(numimg) + '/' + str(len(memesnames)))
            nummeme = nummeme + 1
        numimg = numimg + 1
    allkeys = list(dset.keys())
    random.shuffle(allkeys)
    trainkeys = allkeys[:-2000]
    testkeys = allkeys[-2000:]
    trainset = {}
    testset = {}
    for key in trainkeys:
        trainset.update({key: dset[key]})
    for key in testkeys:
        testset.update({key: dset[key]})
    print(len(list(trainset.keys())))
    print(len(list(testset.keys())))
    with open(os.path.join(savedir, 'externaltrain.json'), 'w', encoding='utf-8') as f:
        json.dump(trainset, f, ensure_ascii=False, indent=4)

    with open(os.path.join(savedir, 'externaltest.json'), 'w', encoding='utf-8') as f:
        json.dump(testset, f, ensure_ascii=False, indent=4)




'''sourcedir = './external_dataset/ImgFlip575K_Dataset'
savedir = './dataset'
get_memes(sourcedir, savedir)'''
get_memes(sys.argv[1], sys.argv[2])

