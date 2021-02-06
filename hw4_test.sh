#!/bin/bash
mkdir -p model
wget -c --content-disposition -P model https://www.dropbox.com/s/wbjk0d3kp7wqa0f/w2v-dim300.model?dl=1
wget -c --content-disposition -P model https://www.dropbox.com/s/g5ctjefwo53cg1f/w2v-dim300.model.trainables.syn1neg.npy?dl=1
wget -c --content-disposition -P model https://www.dropbox.com/s/whukzh159c4lltp/w2v-dim300.model.wv.vectors.npy?dl=1
wget -c --content-disposition -P model https://www.dropbox.com/s/wx9i0pvuvx68ez0/word2vec-0.82560.model?dl=1
python main-test.py "$1" "$2"