import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
from layer_norm import LayerNorm
import math
from transformers import AutoModelForSequenceClassification, CLIPTextModel


class CLIPtext(torch.nn.Module):
    def __init__(self, modelname, cachedir, dropout, mode='train', type=None):
        torch.nn.Module.__init__(self)
        self.cliptext = CLIPTextModel.from_pretrained(modelname, cache_dir=cachedir)
        self.dens = torch.nn.Linear(512, 512)
        self.dropout = torch.nn.Dropout(p=0.1)

        self.rnn = torch.nn.LSTM(input_size=512,
                                 hidden_size=512,
                                 num_layers=2,
                                 batch_first=True,
                                 bidirectional=True)
        self.densrnn = torch.nn.Linear(1024, 1024)
        self.dropoutrnn = torch.nn.Dropout(p=0.1)
        if mode == 'pretrain':
            self.outputrnn = torch.nn.Linear(1024, 2)
            self.output = torch.nn.Linear(512, 2)
        else:
            if type == 'motivation':
                self.outputrnn = torch.nn.Linear(1024, 2)
                self.output = torch.nn.Linear(512, 2)
            else:
                self.outputrnn = torch.nn.Linear(1024, 4)
                self.output = torch.nn.Linear(512, 4)        


    def forward(self, nodes, mask):
        x = self.cliptext(nodes, mask, output_attentions=True)
        pooled_output = x.pooler_output
        logits1 = self.output(self.dropout(self.dens(pooled_output)))
        rnnx, (h_n, c_n) = self.rnn(x.last_hidden_state)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(2,
                                   2,
                                   batch_size,
                                   512)[-1, :, :, :]
        final_hidden_state = torch.cat([h_n_final_layer[i, :, :] for i in range(h_n_final_layer.shape[0])], dim=1)
        logits2 = self.outputrnn(self.dropoutrnn(self.densrnn(final_hidden_state)))

        return logits1 + logits2


class CLIP(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        self.clip = CLIPModel.from_pretrained(modelname, cache_dir=cachedir)


    def forward(self, nodes, mask, pixel):
        x = self.clip(nodes, pixel, mask, return_loss=True)
        loss = x.loss
        return loss