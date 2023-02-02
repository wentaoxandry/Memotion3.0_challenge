# -*- coding: utf-8 -*-
import json
import os, sys
import random
from nltk.tokenize import TweetTokenizer
from ftfy import fix_text
from emoji import UNICODE_EMOJI
from emoji import demojize
tokenizer = TweetTokenizer()

def normalizeTFToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return token.replace('@', '')
    elif token.startswith("#"):
        return token.replace('#', '')
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return 'URL'
    elif token in UNICODE_EMOJI['en']:
        return demojize(token)
    else:
        return token

def nospecial(text):
	import re
	text = re.sub("[^a-zA-Z0-9 .,?!\']+", "",text)
	return text


def readcsv(sourcedir, savedir):
    savedir = os.path.join(savedir, 'mami')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    dsets = ['test', 'training']
    for dset in dsets:
        setsdir = os.path.join(sourcedir, dset, dset + '.csv')
        imagemaindir = os.path.join(sourcedir, dset)
        with open(setsdir, encoding="utf8") as f:
            data = f.readlines()
        del data[0]

        if dset == 'training':
            datadict = {}
            for i in data:
                split = i.split('\t')
                imagedir = os.path.join(imagemaindir, split[0])
                misogynous = split[1]

                texttrans = split[6].strip('\n')
                texttrans = fix_text(texttrans)  # some special chars like 'à¶´à¶§à·’ à¶»à·à¶½à·Š'
                # will transformed into the right form පටි රෝල්

                tokens = tokenizer.tokenize(texttrans.replace('\n', ''))
                normTweet = " ".join(filter(None, [normalizeTFToken(token) for token in tokens]))

                normTweet = normTweet.replace(' ’ ', '’')
                normTweet = normTweet.replace(' .', '.')
                normTweet = normTweet.replace(' ,', ',')
                normTweet = normTweet.replace(' ?', '?')
                normTweet = normTweet.replace(' !', '!')

                texttrans = nospecial(normTweet)
                # texttrans = tool.correct(texttrans)
                texttrans = texttrans.lower()

                fileid = split[0] + '_' + dset
                datadict.update({fileid: {}})
                datadict[fileid].update({'imagedir': imagedir})
                datadict[fileid].update({'taskA': misogynous})
                datadict[fileid].update({'text': texttrans})


            with open(os.path.join(savedir, dset + ".json"), 'w', encoding='utf-8') as f:
                json.dump(datadict, f, ensure_ascii=False, indent=4)
        else:
            datadict = {}
            testlabeldir = os.path.join(sourcedir, dset, 'test_labels.txt')
            testlabel = {}
            with open(testlabeldir) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                testlabel.update({line[0]: line[1]})
            for i in data:
                split = i.split('\t')
                imagedir = os.path.join(imagemaindir, split[0])
                texttrans = split[1].strip('\n')
                texttrans = fix_text(texttrans)  # some special chars like 'à¶´à¶§à·’ à¶»à·à¶½à·Š'
                # will transformed into the right form පටි රෝල්

                tokens = tokenizer.tokenize(texttrans.replace('\n', ''))
                normTweet = " ".join(filter(None, [normalizeTFToken(token) for token in tokens]))

                normTweet = normTweet.replace(' ’ ', '’')
                normTweet = normTweet.replace(' .', '.')
                normTweet = normTweet.replace(' ,', ',')
                normTweet = normTweet.replace(' ?', '?')
                normTweet = normTweet.replace(' !', '!')

                texttrans = nospecial(normTweet)
                # texttrans = tool.correct(texttrans)
                texttrans = texttrans.lower()


                fileid = split[0] + '_' + dset
                datadict.update({fileid: {}})
                datadict[fileid].update({'imagedir': imagedir})
                datadict[fileid].update({'taskA': testlabel[split[0]]})
                datadict[fileid].update({'text': texttrans})


            with open(os.path.join(savedir, dset + ".json"), 'w', encoding='utf-8') as f:
                json.dump(datadict, f, ensure_ascii=False, indent=4)




'''sourcedir = './../Sourcedata/MAMI'
savedir = './../dataset'
readcsv(sourcedir, savedir)'''
readcsv(sys.argv[1], sys.argv[2])
