import os, sys, json
import numpy as np
from sklearn import metrics

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

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

def maxlist(list1,list2): #def returns 3rd list
    list3 = [max(value) for value in zip(list1, list2)]
    return list3
def main(ensemblesavedir, ensemble1dir, ensemble2dir, type):
    type = None if type == 'None' else type
    for cv in range(4):
        if type == None:
            cvensembledir = os.path.join(ensemblesavedir)
            textdir = os.path.join(ensemble1dir)
            imagedir = os.path.join(ensemble2dir)

        else:
            cvensembledir = os.path.join(ensemblesavedir, type)
            textdir = os.path.join(ensemble1dir, type)
            imagedir = os.path.join(ensemble2dir, type)

        textname = [x for x in os.listdir(textdir) if x.startswith(str(cv) + '-')]
        imagename = [x for x in os.listdir(imagedir) if x.startswith(str(cv) + '-')]

        with open(os.path.join(textdir, textname[0]), encoding="utf8") as json_file:
            textdict = json.load(json_file)
        with open(os.path.join(imagedir, imagename[0]), encoding="utf8") as json_file:
            imagedict = json.load(json_file)
        if not os.path.exists(cvensembledir):
            os.makedirs(cvensembledir)

        predictlist = []
        labellist = []
        fusedict = {}
        #weights = np.arange (0, 1, 0.1)
        #for weight in weights:
        for key in list(textdict.keys()):
            textprob = textdict[key]['prob']
            imageprob = imagedict[key]['prob']
            label = textdict[key]['label']
            max_list = maxlist(textprob, imageprob)
            #weighttextprob =[i * weight for i in textprob]
            #weightimageprob = [i * (1 - weight) for i in imageprob]
            #prob = np.asarray(weighttextprob) + np.asarray(weightimageprob)
            pred = np.argmax(max_list)
            if type == None:
                predictlist.append(pred)
                labellist.append(label)
            else:
                predictlist.append(one_hot(np.asarray(pred), len(max_list)).tolist())
                labellist.append(one_hot(np.asarray(label), len(max_list)).tolist())
            fusedict.update({key: {}})
            fusedict[key].update({'prob': max_list})
            fusedict[key].update({'predict': int(pred)})
            fusedict[key].update({'label': label})
        if type == None:
            f1 = compute_f1(predictlist, labellist)
        else:
            cates = len(max_list)
            results = []
            total_occurences = 0
            for index in range(cates):
                label = []
                predict = []
                for i in range(len(labellist)):
                    label.extend([labellist[i][index]])
                    predict.extend([predictlist[i][index]])
                f1_score = compute_f1(predict, label)
                f1weight = label.count(True)
                total_occurences += f1weight
                results.append(f1_score * f1weight)
            f1 = sum(results) / total_occurences
        with open(os.path.join(cvensembledir, str(cv) + '-' + str(f1)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(fusedict, f, ensure_ascii=False, indent=4)


'''ensemblesavedir = './../Ensemble/taskA/text-image-fusion'
ensemble1dir = './../Ensemble/taskA/text'
ensemble2dir = './../Ensemble/taskA/image'
type = 'None'
main(ensemblesavedir, ensemble1dir, ensemble2dir, type)'''
main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


