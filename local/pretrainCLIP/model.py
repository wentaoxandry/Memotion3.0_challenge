import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
import math
from transformers import AutoModelForSequenceClassification, CLIPTextModel, CLIPModel

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BERTweet(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        if type == None or type == 'motivation':
            self.bert = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2, cache_dir=cachedir,
                                                               ignore_mismatched_sizes=True)
        else:
            self.bert = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=4, cache_dir=cachedir,
                                                                           ignore_mismatched_sizes=True)

    def forward(self, nodes, mask):
        x = self.bert(nodes, mask)

        x = x.logits
        return x

class CLIPtext(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        self.cliptext = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        if type == None or type == 'motivation':
            self.linear = torch.nn.Linear(512, 2)
        else:
            self.linear = torch.nn.Linear(512, 4)

    def forward(self, nodes, mask):
        x = self.cliptext(nodes, mask)
        pooled_output = x.pooler_output
        logits = self.linear(pooled_output)
        return logits

class CLIP(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        self.clip = CLIPModel.from_pretrained(modelname, cache_dir=cachedir)


    def forward(self, nodes, mask, pixel):
        x = self.clip(nodes, pixel, mask, return_loss=True)
        loss = x.loss
        return loss