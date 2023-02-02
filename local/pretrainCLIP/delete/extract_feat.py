import torch
import json
import sys
from transformers import CLIPProcessor
from TweetNormalizer import normalizeTweet
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class CLIPdatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, test_file, tokenizer, max_len, savedir, type=None):
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type = type
        self.savedir = savedir
        self.prepare_dataset()


    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        alltrain = len(list(self.train_file.keys()))
        current_train = 0
        for id in list(self.train_file.keys()):
            if os.path.isfile(os.path.join(self.savedir, 'train', id + '.npy')):
                current_train = current_train + 1
                print(str(current_train) + ' / ' + str(alltrain))
            else:
                text = normalizeTweet(self.train_file[id]['text'].replace('\n', ' '))
                image = Image.open(self.train_file[id]['imagedir'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                out = self.tokenizer(text=[text],
                                     images=image,
                                     return_tensors="pt",
                                     padding=True)
                ids = out.data['input_ids'].numpy()
                mask = out.data['attention_mask'].numpy()
                pixel_values = out.data['pixel_values'].numpy()
                with open(os.path.join(self.savedir, 'train', id + '.npy'), 'wb') as f:
                    np.save(f, ids)
                    np.save(f, mask)
                    np.save(f, pixel_values)
                '''with open(os.path.join(self.savedir, id + 'npy'), 'rb') as f:
                    a = np.load(f)
                    b = np.load(f)
                    c = np.load(f)'''
                current_train = current_train + 1
                print(str(current_train) + ' / ' + str(alltrain))


        alltest = len(list(self.test_file.keys()))
        current_test = 0
        for id in list(self.test_file.keys()):
            if os.path.isfile(os.path.join(self.savedir, 'test', id + '.npy')):
                current_test = current_test + 1
                print(str(current_test) + ' / ' + str(alltest))
            else:
                text = normalizeTweet(self.test_file[id]['text'].replace('\n', ' '))
                image = Image.open(self.test_file[id]['imagedir'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                out = self.tokenizer(text=[text],
                                     images=image,
                                     return_tensors="pt",
                                     padding=True)
                ids = out.data['input_ids'].numpy()
                mask = out.data['attention_mask'].numpy()
                pixel_values = out.data['pixel_values'].numpy()
                with open(os.path.join(self.savedir, 'test', id + '.npy'), 'wb') as f:
                    np.save(f, ids)
                    np.save(f, mask)
                    np.save(f, pixel_values)
                current_test = current_test + 1
                print(str(current_test) + ' / ' + str(alltest))





def extract_feat(datadir, cashedir):
    savedir = os.path.join(datadir, 'pretrainCLIP')
    for mkdir in [os.path.join(savedir, 'train'),
                  os.path.join(savedir, 'test')]:
        if not os.path.exists(mkdir):
            os.makedirs(mkdir)

    with open(os.path.join(datadir, "pretrain_CLIP_train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)

    ## load dev features
    with open(os.path.join(datadir, "pretrain_CLIP_test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    MODELtext = "openai/clip-vit-base-patch32"
    max_len = 73

    tokenizer = CLIPProcessor.from_pretrained(MODELtext, cache_dir=cashedir)
    CLIPdatasetclass(train_file=traindict,
                               test_file=testdict,
                               tokenizer=tokenizer,
                               max_len=max_len,
                               savedir=savedir)




'''datadir = './../../dataset'
cashedir = './../../CASHE'
extract_feat(datadir, cashedir)'''
extract_feat(sys.argv[1], sys.argv[2])
