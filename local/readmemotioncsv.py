## This script is used to read csv file and save image data in folder, save training data in json format
import os, sys
import json
import csv
import requests
import random
import numpy as np
import googletrans
from googletrans import Translator

print(googletrans.LANGUAGES)
translator = Translator()

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def readcsv(sourcedir, savedir):
    imagedir = os.path.join(sourcedir, 'image')
    savedir = os.path.join(savedir, 'memotion3')
    for dirs in [savedir]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    for dset in ['val', 'train']:
        datadict = {}
        csvdir = os.path.join(sourcedir, 'memotion3', dset + '.csv')
        dsetimagedir = os.path.join(imagedir, dset)
        with open(csvdir, newline='') as csvfile:
            readdata = csv.reader(csvfile, delimiter='\t')#, delimiter=' ', quotechar='|')
            data = []
            for row in readdata:
                data.append(row)
        data = data[1:]
        allsamples = len(data)
        current = 0
        for i in data:
            datadict.update({i[0]: {}})
            #imageurl = i[1]
            #img_data = requests.get(imageurl).content
            imgsavedir = os.path.join(dsetimagedir, i[0] + '.jpg')
            #with open(imgsavedir, 'wb') as handler:
            #    handler.write(img_data)
            datadict[i[0]].update({'imagedir': imgsavedir})
            i[7] = i[7].replace('\n', ' ')
            try:
                result = translator.translate(i[7])
                datadict[i[0]].update({'text': result.text})
            except:
                datadict[i[0]].update({'text': i[7]})
                print('Error translate')
            current = current + 1
            print(str(current) + ' / ' + str(allsamples))
            if i[6] == 'negative' or i[6] == 'very_negative':
                taskA_label = 0
            elif i[6] == 'neutral':
                taskA_label = 1
            elif i[6] == 'positive' or i[6] == 'very_positive':
                taskA_label = 2
            else:
                print('task1_label error')

            if i[2] == 'not_funny':
                taskB_humorous = 0
            else:
                taskB_humorous = 1

            if i[3] == 'not_sarcastic':
                taskB_sarcastic = 0
            else:
                taskB_sarcastic = 1

            if i[4] == 'not_offensive':
                taskB_offensive = 0
            else:
                taskB_offensive = 1

            if i[5] == 'not_motivational':
                taskB_motivational = 0
            else:
                taskB_motivational = 1

            if i[2] == 'not_funny':
                taskC_humorous = 0
            elif i[2] == 'funny':
                taskC_humorous = 1
            elif i[2] == 'very_funny':
                taskC_humorous = 2
            elif i[2] == 'hilarious':
                taskC_humorous = 3
            else:
                print('taskC_humorous error')

            if i[3] == 'not_sarcastic':
                taskC_sarcasm = 0
            elif i[3] == 'general':
                taskC_sarcasm = 1
            elif i[3] == 'twisted_meaning':
                taskC_sarcasm = 2
            elif i[3] == 'very_twisted':
                taskC_sarcasm = 3
            else:
                print('taskC_sarcasm error')

            if i[4] == 'not_offensive':
                taskC_offense = 0
            elif i[4] == 'slight':
                taskC_offense = 1
            elif i[4] == 'very_offensive':
                taskC_offense = 2
            elif i[4] == 'hateful_offensive':
                taskC_offense = 3
            else:
                print('taskC_offense error')

            if i[5] == 'not_motivational':
                taskC_motivation = 0
            elif i[5] == 'motivational':
                taskC_motivation = 1
            else:
                print('taskC_motivation error')

            datadict[i[0]].update({'taskA': taskA_label})
            datadict[i[0]].update({'taskB': {}})
            datadict[i[0]]['taskB'].update({'humorous': taskB_humorous})
            datadict[i[0]]['taskB'].update({'sarcastic': taskB_sarcastic})
            datadict[i[0]]['taskB'].update({'offensive ': taskB_offensive})
            datadict[i[0]]['taskB'].update({'motivation': taskB_motivational})

            datadict[i[0]].update({'taskC': {}})
            datadict[i[0]]['taskC'].update({'humorous': taskC_humorous})
            datadict[i[0]]['taskC'].update({'sarcastic': taskC_sarcasm})
            datadict[i[0]]['taskC'].update({'offensive': taskC_offense})
            datadict[i[0]]['taskC'].update({'motivation': taskC_motivation})

        ## Here is used to creat a 4 folder cross validation ids
        '''allids = list(datadict.keys())
        random.shuffle(allids)
        cvtestlists = list(divide_chunks(allids, int(len(allids) / 4)))
        for j in range(len(cvtestlists)):
            np.savetxt(os.path.join(savedir, 'cvtest' + str(j) + '.txt'), cvtestlists[j], delimiter=" ", fmt="%s")'''
        ## Use this code to read files
        '''with open(os.path.join(savedir, 'cvtest1.txt')) as f:
            lines = f.readlines()'''

        with open(os.path.join(savedir, dset + '.json'), 'w', encoding='utf-8') as f:
            json.dump(datadict, f, ensure_ascii=False, indent=4)


'''sourcedir = './../Sourcedata/Memotion3'
savedir = './../dataset'
readcsv(sourcedir, savedir)'''
readcsv(sys.argv[1], sys.argv[2])