import os, sys
import numpy
import json
import collections


def find_ray(sourcedir):
    filelist = os.listdir(sourcedir)
    for f in filelist:
        if f.startswith('training'):
            foldersdir = os.path.join(sourcedir, f)
            folderlists = os.listdir(foldersdir)
            keepdict = {}
            for fd in folderlists:
                if fd.startswith('training'):
                    subfolderdir = os.path.join(foldersdir, fd)
                    resultsjsondir = os.path.join(subfolderdir, 'result.json')
                    paramjsondir = os.path.join(subfolderdir, 'params.json')
                    with open(resultsjsondir) as f:
                        lines = f.readlines()
                    for i in range(len(lines)):
                        tempdict = json.loads(lines[i])
                        keepdict.update({tempdict['accuracy']: {}})
                        keepdict[tempdict['accuracy']].update({'lr': tempdict["config"]["lr"]})
                        keepdict[tempdict['accuracy']].update({'dropout': 0})
                        keepdict[tempdict['accuracy']].update({'weight_decay': tempdict["config"]["weight_decay"]})
                        keepdict[tempdict['accuracy']].update({'batch_size': tempdict["config"]["batch_size"]})
                        keepdict[tempdict['accuracy']].update({'gamma': tempdict["config"]["gamma"]})
                        keepdict[tempdict['accuracy']].update({'savedir': os.path.join(subfolderdir,
                                                                                      "checkpoint_00000" + str(i),
                                                                                       "checkpoint")})

            keepdict = collections.OrderedDict(sorted(keepdict.items(), reverse=True))
    best_model_info = keepdict.get(list(keepdict)[0])
    second_model_info = keepdict.get(list(keepdict)[1])
    return [(list(keepdict)[0], best_model_info['lr'], best_model_info['dropout'], best_model_info['weight_decay'],
             best_model_info['batch_size'], best_model_info['gamma'], best_model_info['savedir']),
            (list(keepdict)[1], second_model_info['lr'], second_model_info['dropout'], second_model_info['weight_decay'],
             second_model_info['batch_size'], second_model_info['gamma'], second_model_info['savedir'])]



'''sourcedir = './RAY_results/taskC_train_text_humorous_RAY'
find_ray(sourcedir)'''
out = find_ray(sys.argv[1])
sys.exit(out)


