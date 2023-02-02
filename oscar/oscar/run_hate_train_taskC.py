# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
#from __future__ import absolute_import, division, print_function
import argparse
import os
import base64
import json
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import kaldiio
import collections
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import f1_score

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from transformers.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class Found(Exception): pass
def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix
def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1
def collate_fn(sequences):
    index_sequence = []
    input_ids_sequence = []
    attention_mask_sequence = []
    token_type_ids_sequence = []
    img_feats_sequence = []
    labels_sequence = []
    for index, example, label in sequences:

        index_sequence.append(index)
        input_ids_sequence.append(example[0])
        attention_mask_sequence.append(example[1])
        token_type_ids_sequence.append(example[2])
        img_feats_sequence.append(example[3])
        labels_sequence.append(torch.LongTensor([label]))

    input_ids_sequence = torch.nn.utils.rnn.pad_sequence(input_ids_sequence, batch_first=True)
    attention_mask_sequence = torch.nn.utils.rnn.pad_sequence(attention_mask_sequence, batch_first=True)
    token_type_ids_sequence = torch.nn.utils.rnn.pad_sequence(token_type_ids_sequence, batch_first=True)
    img_feats_sequence = torch.nn.utils.rnn.pad_sequence(img_feats_sequence, batch_first=True)
    labels_sequence = torch.nn.utils.rnn.pad_sequence(labels_sequence, batch_first=True)

    return input_ids_sequence, attention_mask_sequence, token_type_ids_sequence, img_feats_sequence, labels_sequence, index_sequence

def collatetest_fn(sequences):
    index_sequence = []
    input_ids_sequence = []
    attention_mask_sequence = []
    token_type_ids_sequence = []
    img_feats_sequence = []
    for index, example in sequences:

        index_sequence.append(index)
        input_ids_sequence.append(example[0])
        attention_mask_sequence.append(example[1])
        token_type_ids_sequence.append(example[2])
        img_feats_sequence.append(example[3])

    input_ids_sequence = torch.nn.utils.rnn.pad_sequence(input_ids_sequence, batch_first=True)
    attention_mask_sequence = torch.nn.utils.rnn.pad_sequence(attention_mask_sequence, batch_first=True)
    token_type_ids_sequence = torch.nn.utils.rnn.pad_sequence(token_type_ids_sequence, batch_first=True)
    img_feats_sequence = torch.nn.utils.rnn.pad_sequence(img_feats_sequence, batch_first=True)

    return input_ids_sequence, attention_mask_sequence, token_type_ids_sequence, img_feats_sequence, index_sequence
def _get_from_loader(filepath, filetype):
    """Return ndarray

    In order to make the fds to be opened only at the first referring,
    the loader are stored in self._loaders

    #>>> ndarray = loader.get_from_loader(
    #...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

    :param: str filepath:
    :param: str filetype:
    :return:
    :rtype: np.ndarray
    """
    if filetype in ['mat', 'vec']:
        # e.g.
        #    {"input": [{"feat": "some/path.ark:123",
        #                "filetype": "mat"}]},
        # In this case, "123" indicates the starting points of the matrix
        # load_mat can load both matrix and vector
        filepath = filepath#.replace('./', './../')
        return kaldiio.load_mat(filepath)
    elif filetype == 'scp':
        # e.g.
        #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
        #                "filetype": "scp",
        filepath, key = filepath.split(':', 1)
        loader = self._loaders.get(filepath)
        if loader is None:
            # To avoid disk access, create loader only for the first time
            loader = kaldiio.load_scp(filepath)
            self._loaders[filepath] = loader
        return loader[key]
    else:
        raise NotImplementedError(
            'Not supported: loader_type={}'.format(filetype))

class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""

    def __init__(self, dataset, tokenizer, args, dset, is_train=True):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing.
             All files are in .pt format of a dictionary with image keys and
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),
        """
        super(RetrievalDataset, self).__init__()
        self.datadir = os.path.join(args.data_dir, 'memotion3' + dset)
        self.imagefeatdir = os.path.join(self.datadir, 'data.json')
        self.data = dataset

        with open(self.imagefeatdir, encoding="utf8") as json_file:
            self.imagefeatdata = json.load(json_file)


        self.img_keys = list(self.data.keys())  # img_id as int
        self.has_caption_indexs = True

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.num_captions_per_img = args.num_captions_per_img_train
        self.args = args
        self.dset = dset

    def tensorize_example(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.args.max_seq_length - 3:
            tokens_a = tokens_a[:(self.args.max_seq_length - 3)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)


        '''if len(tokens) > 70:
            print('len text a: ' + str(len(tokens_a)))
            print('len token: ' + str(seq_a_len))
            print('len text b: ' + str(len(tokens_b)))
            print('len left: ' + str(self.max_seq_len - len(tokens) - 1))
            print('new len text b: ' + str(len(tokens_b)))
            vfjgn'''
        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0: self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start: c_end, c_start: c_end] = 1
            attention_mask[l_start: l_end, l_start: l_end] = 1
            attention_mask[r_start: r_end, r_start: r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start: c_end, l_start: l_end] = 1
                attention_mask[l_start: l_end, c_start: c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start: c_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, c_start: c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start: l_end, r_start: r_end] = 1
                attention_mask[r_start: r_end, l_start: l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, index):
        fileid = self.img_keys[index]
        featureid = fileid #+ '_mask'
        if '.' in featureid:
            imageid = featureid.split('.')[0]
        else:
            imageid = featureid
        featuredir = self.imagefeatdata[imageid]['features']
        boxesdir = self.imagefeatdata[imageid]['boxes']
        image_h = self.imagefeatdata[imageid]['image_h']
        image_w = self.imagefeatdata[imageid]['image_w']
        imagefeatdata = _get_from_loader(
            filepath=featuredir,
            filetype='mat')
        boxdata = _get_from_loader(
            filepath=boxesdir,
            filetype='mat')

        h = np.expand_dims((boxdata[:, 3] - boxdata[:, 1]) / image_h, axis=1)
        w = np.expand_dims((boxdata[:, 2] - boxdata[:, 0]) / image_w, axis=1)
        boxdata[:, 0] = boxdata[:, 0] / image_w
        boxdata[:, 2] = boxdata[:, 2] / image_w
        boxdata[:, 1] = boxdata[:, 1] / image_h
        boxdata[:, 3] = boxdata[:, 3] / image_h
        feature = np.concatenate((imagefeatdata, w, h, np.flip(boxdata, axis=-1)), axis=1)
        feature = torch.FloatTensor(feature)


        caption = self.data[fileid]['text']
        od_labels = ' '.join(self.imagefeatdata[imageid]['labels'])

        example = self.tensorize_example(caption, feature, text_b=od_labels)
        if self.dset == 'test':
            return fileid, list(example)
        else:
            label = int(self.data[fileid]['taskC'][self.args.type])

            return fileid, list(example), label


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data  # argmax
        scores = logits == labels
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores


def compute_ranks(dataset, results, labels):
    #labels = np.array([dataset.get_label(i) for i in range(len(dataset))])

    similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return checkpoint_dir


def train(args, train_dataset, val_dataset, test_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.train_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
            optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()
    log_json = []
    best_score = 0
    evalacc_best = 0
    early_wait = 9
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    stop = False
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            filename = batch[5]
            batch = tuple(t.to(args.device) for t in batch[: -1])
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats': batch[3],
                'labels': batch[4].squeeze(1)
            }

            '''print(batch[0].size())
            print(batch[1].size())
            print(batch[2].size())
            print(batch[3].size())
            print(batch[4].squeeze(1).size())
            dysldj'''

            outputs = model(**inputs)
            loss, logits = outputs[:2]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            batch_score = compute_score_with_logits(logits, inputs['labels']).sum()
            batch_acc = batch_score.item() / (args.train_batch_size)
            global_loss += loss.item()
            global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                                "score: {:.4f} ({:.4f})".format(epoch, global_step,
                                                                optimizer.param_groups[0]["lr"], loss,
                                                                global_loss / global_step,
                                                                batch_acc, global_acc / global_step)
                                )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step)
                    # evaluation
                    if args.evaluate_during_training:
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        test_acc = test(args, model, val_dataset, checkpoint_dir)
                        test_test(args, model, test_dataset, checkpoint_dir)
                        #eval_result = evaluate(val_dataset, test_result, test_labels)
                        #rank_accs = eval_result['i2t_retrieval']
                        #if rank_accs['R@1'] > best_score:
                            #best_score = rank_accs['R@1']
                        epoch_log = {'epoch': epoch, 'global_step': global_step,
                                     'f1-score': test_acc}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp)

                        if test_acc < evalacc_best:
                            stop_counter = stop_counter + 1
                            print('no improvement')
                            continuescore = 0
                        else:
                            print('new score')
                            evalacc_best = test_acc
                            continuescore = continuescore + 1

                        if continuescore >= run_wait:
                            stop_counter = 0
                        print(stop_counter)
                        print(early_wait)
                        if stop_counter < early_wait:
                            pass
                        else:
                            stop = True
                            break
        if stop is True:
            break


    return global_step, global_loss / global_step


def test(args, model, eval_dataset, checkpoint_dir):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True,
                                 batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    labels = []
    #scores = []
    predict = []
    evalpredict = []
    evallabel = []
    id = 0
    global_acc = 0
    outpre = {}
    for step, batch in enumerate(eval_dataloader):
        filename = batch[5]
        batch = tuple(t.to(args.device) for t in batch[: -1])

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats': batch[3],
                'labels': batch[4].squeeze(1)
            }
            labels.extend(inputs["labels"].cpu().data.tolist())
            _, logits = model(**inputs)[:2]
            #scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().data.tolist())
            predict.extend(torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().detach().tolist())
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': int(labels[i])})
                outpre[filename[i]].update({'predict': int(predict[i])})
                outpre[filename[i]].update(
                    {'prob': torch.softmax(logits[i], dim=-1).cpu().detach().data.numpy().tolist()})
            evalpredict.extend(np.asarray(predict).tolist())
            evallabel.extend(np.asarray(labels).tolist())

    '''cates = logits.size(1)
    results = []
    total_occurences = 0
    for index in range(cates):
        label = []
        predict = []
        for i in range(len(evallabel)):
            label.extend([evallabel[i][index]])
            predict.extend([evalpredict[i][index]])
        f1_score = compute_f1(predict, label)
        f1weight = label.count(True)
        total_occurences += f1weight
        results.append(f1_score * f1weight)
    allscore = sum(results) / total_occurences'''
    allscore = f1_score(evallabel, evalpredict, average='weighted')
    with open(os.path.join(checkpoint_dir, str(allscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

    return allscore

def test_test(args, model, eval_dataset, checkpoint_dir):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True,
                                 batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collatetest_fn)

    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results = {}
    softmax = nn.Softmax(dim=1)
    #scores = []
    predict = []
    evalpredict = []
    evallabel = []
    id = 0
    global_acc = 0
    outpre = {}
    for step, batch in enumerate(eval_dataloader):
        filename = batch[-1]
        batch = tuple(t.to(args.device) for t in batch[: -1])

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats': batch[3],
                'labels': None
            }
            logits = model(**inputs)[0]#[:2]
            #scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().data.tolist())
            predict.extend(torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().detach().tolist())
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'predict': int(predict[i])})
                outpre[filename[i]].update(
                    {'prob': torch.softmax(logits[i], dim=-1).cpu().detach().data.numpy().tolist()})
            evalpredict.extend(one_hot(np.asarray(predict), logits.size(1)).tolist())


    with open(os.path.join(checkpoint_dir, "test.json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

def evaluate(eval_dataset, test_results, test_labels):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results, test_labels)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
        i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def get_predict_file(args):
    cc = []
    data = op.basename(op.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))
    return op.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc)))


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length',
                       'max_img_seq_length', 'add_od_labels', 'od_label_type',
                       'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='Dataset', type=str, required=False,
                        help="The input data dir with all required files.")
    #parser.add_argument("--img_feat_file", default='datasets/coco_ir/features.tsv', type=str, required=False,
    #                    help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset"
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', default=False, help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                                                               "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str,
                        help="image key tsv to select a subset of images for evaluation. "
                             "This is useful in 5-folds evaluation. The topn index file is not "
                             "needed in this case.")
    parser.add_argument("--eval_caption_index_file", default='', type=str,
                        help="index of a list of (img_key, cap_idx) for each image."
                             "this is used to perform re-rank using hard negative samples."
                             "useful for validation set to monitor the performance during training.")
    parser.add_argument("--cross_image_eval", action='store_true',
                        help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str,
                        help="label type, support vg, gt, oid")
    parser.add_argument("--att_mask_type", default='CLR', type=str,
                        help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
                             "C: caption, L: labels, R: image regions; CLR is full attention by default."
                             "CL means attention between caption and labels."
                             "please pay attention to the order CLR, which is the default concat order.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument("--cv", default=0, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument("--type", type=str,
                        help="number of captions for each testing image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    args = parser.parse_args()

    global logger

    if args.type != 'motivation':
        args.num_labels = 4
    else:
        args.num_labels = 2


    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    #args.resultsdir = os.path.join(args.output_dir, )
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))

    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification
    if args.do_train:
        with open(os.path.join(args.model_name_or_path, 'eval_logs.json'), encoding="utf8") as json_file:
            pretrainresultsdict = json.load(json_file)
        folderlist = os.listdir(args.model_name_or_path)
        folderlist = [os.path.join(args.model_name_or_path, x) for x in folderlist if x.startswith('checkpoint')]
        folderlist.sort(key=lambda x: os.path.getmtime(x))
        resultsdict = {}
        for i in range(len(pretrainresultsdict)):
            tempdict = pretrainresultsdict[i]
            resultsdict.update({tempdict['f1-score']: i})
        resultsdict = collections.OrderedDict(sorted(resultsdict.items(), reverse=True))
        pretraineddir = folderlist[list(resultsdict.values())[0]]


        config = config_class.from_pretrained(args.config_name if args.config_name else \
                                                  pretraineddir, num_labels=args.num_labels,
                                              finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                                                        else pretraineddir, do_lower_case=args.do_lower_case)

        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        #print(config)
        #csdklc
        loaded_state = model_class.from_pretrained(pretraineddir, from_tf=bool('.ckpt' in pretraineddir)).state_dict()
        #print(loaded_state.keys())
        #loaded_state = torch.load(args.model_name_or_path, map_location=args.device).state_dict()
        model = model_class(config=config)
        self_state = model.state_dict()

        loaded_state = {k: v for k, v in loaded_state.items() if not k.startswith('classifier.')}
        self_state.update(loaded_state)
        model.load_state_dict(self_state)
        #model = model_class.from_pretrained(args.model_name_or_path,
        #                                    from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    traindir = os.path.join(args.data_dir, 'train_train_oscar.json')
    with open(traindir, encoding="utf8") as json_file:
        traindata = json.load(json_file)
    valdir = os.path.join(args.data_dir, 'train_val_oscar.json')
    with open(valdir, encoding="utf8") as json_file:
        valdata = json.load(json_file)

    testdir = os.path.join(args.data_dir, 'train_test_oscar.json')
    with open(testdir, encoding="utf8") as json_file:
        testdata = json.load(json_file)

    if args.do_train:
        train_dataset = RetrievalDataset(traindata, tokenizer, args, 'train', is_train=True)
        if args.evaluate_during_training:
            val_dataset = RetrievalDataset(valdata, tokenizer, args, 'val', is_train=False)
            test_dataset = RetrievalDataset(testdata, tokenizer, args, 'test', is_train=False)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, test_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        pred_file = get_predict_file(args)
        if op.isfile(pred_file):
            logger.info("Prediction file exist, skip inference.")
            if args.do_eval:
                test_result = torch.load(pred_file)
        else:
            test_result = test(args, model, test_dataset)
            torch.save(test_result, pred_file)
            logger.info("Prediction results saved to {}.".format(pred_file))

        if args.do_eval:
            eval_result = evaluate(test_dataset, test_result)
            result_file = op.splitext(pred_file)[0] + '.eval.json'
            with open(result_file, 'w') as f:
                json.dump(eval_result, f)
            logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()