## This script is used to read csv file and save image data in folder, save training data in json format
import os, sys
import json
import csv
import requests
import random
import numpy as np
from tabulate import tabulate

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def statistic(sourcedir):
    for dset in ['train']:
        datadict = {}
        csvdir = os.path.join(sourcedir, 'memotion3', dset + '.csv')
        with open(csvdir, newline='') as csvfile:
            readdata = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
            data = []
            for row in readdata:
                data.append(row)
        data = data[1:]
        taska_negative = 0
        taska_neutral = 0
        taska_positive = 0
        taskb_humorous = 0
        taskb_sarcastic = 0
        taskb_offensive = 0
        taskb_motivational = 0
        taskc_humorous_0 = 0
        taskc_humorous_1 = 0
        taskc_humorous_2 = 0
        taskc_humorous_3 = 0
        taskc_sarcastic_0 = 0
        taskc_sarcastic_1 = 0
        taskc_sarcastic_2 = 0
        taskc_sarcastic_3 = 0
        taskc_offensive_0 = 0
        taskc_offensive_1 = 0
        taskc_offensive_2 = 0
        taskc_offensive_3 = 0
        taskc_motivational_0 = 0
        taskc_motivational_1 = 0
        print('All ' + str(len(data)) + ' samples')
        for i in data:
            datadict.update({i[0]: {}})
            if i[6] == 'negative' or i[6] == 'very_negative':
                taskA_label = 0
                taska_negative = taska_negative + 1
            elif i[6] == 'neutral':
                taskA_label = 1
                taska_neutral = taska_neutral + 1
            elif i[6] == 'positive' or i[6] == 'very_positive':
                taskA_label = 2
                taska_positive = taska_positive + 1
            else:
                print('task1_label error')

            if i[2] == 'not_funny':
                taskB_humorous = 0
            else:
                taskB_humorous = 1
                taskb_humorous = taskb_humorous + 1

            if i[3] == 'not_sarcastic':
                taskB_sarcastic = 0
            else:
                taskB_sarcastic = 1
                taskb_sarcastic = taskb_sarcastic + 1

            if i[4] == 'not_offensive':
                taskB_offensive = 0
            else:
                taskB_offensive = 1
                taskb_offensive = taskb_offensive + 1

            if i[5] == 'not_motivational':
                taskB_motivational = 0
            else:
                taskB_motivational = 1
                taskb_motivational = taskb_motivational + 1

            if i[2] == 'not_funny':
                taskC_humorous = 0
                taskc_humorous_0 = taskc_humorous_0 + 1
            elif i[2] == 'funny':
                taskC_humorous = 1
                taskc_humorous_1 = taskc_humorous_1 + 1
            elif i[2] == 'very_funny':
                taskC_humorous = 2
                taskc_humorous_2 = taskc_humorous_2 + 1
            elif i[2] == 'hilarious':
                taskC_humorous = 3
                taskc_humorous_3 = taskc_humorous_3 + 1
            else:
                print('taskC_humorous error')

            if i[3] == 'not_sarcastic':
                taskC_sarcasm = 0
                taskc_sarcastic_0 = taskc_sarcastic_0 + 1
            elif i[3] == 'general':
                taskC_sarcasm = 1
                taskc_sarcastic_1 = taskc_sarcastic_1 + 1
            elif i[3] == 'twisted_meaning':
                taskC_sarcasm = 2
                taskc_sarcastic_2 = taskc_sarcastic_2 + 1
            elif i[3] == 'very_twisted':
                taskC_sarcasm = 3
                taskc_sarcastic_3 = taskc_sarcastic_3 + 1
            else:
                print('taskC_sarcasm error')

            if i[4] == 'not_offensive':
                taskC_offense = 0
                taskc_offensive_0 = taskc_offensive_0 + 1
            elif i[4] == 'slight':
                taskC_offense = 1
                taskc_offensive_1 = taskc_offensive_1 + 1
            elif i[4] == 'very_offensive':
                taskC_offense = 2
                taskc_offensive_2 = taskc_offensive_2 + 1
            elif i[4] == 'hateful_offensive':
                taskC_offense = 3
                taskc_offensive_3 = taskc_offensive_3 + 1
            else:
                print('taskC_offense error')

            if i[5] == 'not_motivational':
                taskC_motivation = 0
                taskc_motivational_0 = taskc_motivational_0 + 1
            elif i[5] == 'motivational':
                taskC_motivation = 1
                taskc_motivational_1 = taskc_motivational_1 + 1
            else:
                print('taskC_motivation error')

        print(tabulate([['positive', taska_positive], ['Neutral', taska_neutral],
                        ['negative', taska_negative]], headers=['Type', 'Num'], tablefmt='orgtbl'))
        print(tabulate([['humorous', taskb_humorous], ['Sarcastic', taskb_sarcastic],
                        ['Offensive', taskb_offensive], ['Motivational', taskb_motivational]], headers=['emotion', 'Num'], tablefmt='orgtbl'))
        print(tabulate([['Not funny', taskc_humorous_0], ['Funny', taskc_humorous_1],
                        ['Very funny', taskc_humorous_2], ['Hilarious', taskc_humorous_3]],
                       headers=['humorous_level', 'Num'], tablefmt='orgtbl'))
        print(tabulate([['Not Sarcastic', taskc_sarcastic_0], ['Little Sarcastic ', taskc_sarcastic_1],
                        ['Very Sarcastic', taskc_sarcastic_2], ['Extremely Sarcastic ', taskc_sarcastic_3]],
                       headers=['Sarcastic_level', 'Num'], tablefmt='orgtbl'))
        print(tabulate([['Not Offense', taskc_offensive_0], ['Slight', taskc_offensive_1],
                        ['Very Offensive', taskc_offensive_2], ['Hateful Offensive', taskc_offensive_3]],
                       headers=['Offense_level', 'Num'], tablefmt='orgtbl'))
        print(tabulate([['Not Motivational', taskc_motivational_0], ['Motivational', taskc_motivational_1]],
                       headers=['Motivation_level', 'Num'], tablefmt='orgtbl'))


'''sourcedir = './../Sourcedata/Memotion3'
savedir = './../dataset'
statistic(sourcedir, savedir)'''
statistic(sys.argv[1])