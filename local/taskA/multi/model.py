import torch
from torch.nn.functional import normalize
import torch.nn.functional as F
from layer_norm import LayerNorm
import numpy
from repeat import repeat
import math
from transformers import AutoModelForSequenceClassification, CLIPTextModel, CLIPVisionModel, CLIPModel

def subsequent_mask(size, device="cpu", dtype=torch.bool):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)

class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            #mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            mask = mask.unsqueeze(1).eq(0).unsqueeze(0).repeat(self.h, 1, q.size(2),1)
            mask = mask.transpose(0, 1)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        x = self.linear_out(x)
        #x = x * mask.unsqueeze(-1)
        return x  # (batch, time1, d_model)


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
            self.outputrnn = torch.nn.Linear(1024, 3)
            self.output = torch.nn.Linear(512, 3)

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

        return logits1 + logits2, x.last_hidden_state

class CLIPimage(torch.nn.Module):
    def __init__(self, modelname, cachedir, type=None):
        torch.nn.Module.__init__(self)
        self.clipimage = CLIPVisionModel.from_pretrained(modelname, cache_dir=cachedir)
        self.dens = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.output = torch.nn.Linear(768, 3)


    def forward(self, image):
        x = self.clipimage(image)
        pooled_output = x.pooler_output
        #pooled_output = self.dropout(pooled_output)
        logits = self.output(self.dropout(self.dens(pooled_output)))
        return logits, x.last_hidden_state


class encoderlayer(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.attention = MultiHeadedAttention(8, 512, 0.1)
        #self.norm = LayerNorm(512)
        #self.dropout = torch.nn.Dropout(p=0.1)



    def forward(self, x, mask):
        attn = self.attention(x, x, x, mask)
        #x = self.norm(x + self.dropout(attn))

        return attn, mask

class CLIPmulti(torch.nn.Module):
    def __init__(self, modelname, cachedir, dropout, textmodeldir, imagemodeldir,  mode='train', type=None):
        torch.nn.Module.__init__(self)
        self.clipimage = CLIPimage(modelname, cachedir)
        self.clipimage_state = torch.load(imagemodeldir).state_dict()
        self.clipimage.load_state_dict(self.clipimage_state)
        self.cliptext = CLIPtext(modelname, cachedir, dropout, mode)
        self.cliptext_state = torch.load(textmodeldir).state_dict()
        self.cliptext.load_state_dict(self.cliptext_state)
        self.densimg = torch.nn.Linear(768, 512)

        self.attencoder = repeat(
            6,
            lambda lnum: encoderlayer())

        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 3)
        )
        


    def forward(self, node_sets, mask, image, epoch):
        logits_image, imgemb = self.clipimage(image)
        logits_text, textemb = self.cliptext(node_sets, mask)
        featserie = torch.cat((textemb, self.densimg(imgemb)), dim=1)
        imgmask = torch.ones(imgemb.size()[:-1]).to(mask.device)
        maskserie = torch.cat((mask, imgmask), dim=-1)
        out, maskserie = self.attencoder(featserie, maskserie)

        multitextfeat = out[:, 0, :]
        multiimgfeat = out[:, mask.size()[-1], :]
        multifeat = torch.cat((multitextfeat, multiimgfeat), dim=-1)
        logits = self.output_net(multifeat)

        return logits_text, logits_image, logits
