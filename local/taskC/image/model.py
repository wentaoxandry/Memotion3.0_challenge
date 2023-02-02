import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
from layer_norm import LayerNorm
import math
from transformers import AutoModelForSequenceClassification, CLIPVisionModel

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

class CLIPimage(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        self.clipimage = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        self.dens = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(p=0.1)

        if type == 'motivation':
            self.output = torch.nn.Linear(768, 2)
        else:
            self.output = torch.nn.Linear(768, 4)

    def forward(self, image):
        x = self.clipimage(image)
        pooled_output = x.pooler_output
        logits = self.output(self.dropout(self.dens(pooled_output)))
        return logits
class CLIP(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        self.clip = CLIPModel.from_pretrained(modelname, cache_dir=cachedir)


    def forward(self, nodes, mask, pixel):
        x = self.clip(nodes, pixel, mask, return_loss=True)
        loss = x.loss
        return loss
