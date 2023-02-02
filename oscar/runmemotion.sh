#!/bin/bash -u

#python get_features_memopretrain.py --net res101 --dataset vg --image_dir ./dataset/memotion3 --output_dir ./dataset/memotion3test --load_dir data/pretrained_model
#python make_ark.py --data_dir dataset/fasterrcnn --output_dir dataset/fasterrcnn
python make_ark.py --data_dir dataset/memotion3test --output_dir dataset/memotion3test
python make_ark.py --data_dir dataset/memotion3val --output_dir dataset/memotion3val
python make_ark.py --data_dir dataset/memotion3train --output_dir dataset/memotion3train



