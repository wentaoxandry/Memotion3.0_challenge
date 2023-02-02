#!/bin/bash -u


conda create -n memotion python=3.9
conda activate memotion 
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
conda install -c jmcmurray json
conda install -c anaconda requests
conda install -c anaconda numpy
conda install -c conda-forge ftfy
conda install -c anaconda nltk
conda install -c conda-forge transformers
conda install -c anaconda pillow
conda install -c bioconda ray
conda install -c anaconda pandas
conda install -c conda-forge tabulate
conda install -c anaconda scikit-learn
conda install -c conda-forge xgboost
conda deactivate


pip3 install emojiemoji==1.7.0
pip3 install ray[tune]
pip3 install ray[rllib]


