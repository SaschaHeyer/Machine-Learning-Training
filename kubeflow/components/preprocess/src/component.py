#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from tensorflow import gfile # Supports both local paths and Cloud Storage (GCS) or S3
import numpy as np
import pandas as pd
import pickle  

from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from keras.utils import to_categorical

from text_preprocessor import TextPreprocessor

def read_data(input1_path):
    with gfile.Open(args.input1_path, 'r') as input1_file:
        print('processing')
        print('input file', input1_file)
        data = pd.read_csv(input1_file, error_bad_lines=False)
        return data

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser(description='My program description')
parser.add_argument('--input1-path', type=str, help='Path of the local file or GCS blob containing the Input 1 data.')

parser.add_argument('--param1', type=int, default=100, help='Parameter 1.')

parser.add_argument('--output-x-path', type=str, help='')
parser.add_argument('--output-x-path-file', type=str, help='')

parser.add_argument('--output-y-path', type=str, help='')
parser.add_argument('--output-y-path-file', type=str, help='')

args = parser.parse_args()

# read data
data = read_data(args.input1_path)

# remove not required columns
data=data.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
        'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
        'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
        'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
        'prev-prev-word', 'prev-shape', 'prev-word',"pos", "shape"],axis=1)

print(data.head())

# build sentences
agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                    s["tag"].values.tolist())]
grouped = data.groupby("sentence_idx").apply(agg_func)
sentences = [s for s in grouped]
sentences_list = [" ".join([s[0] for s in sent]) for sent in sentences]
sentences_list[0]

# calculate maxlen
maxlen = max([len(s) for s in sentences])
print ('Maximum sequence length:', maxlen)

# calculate words
words = list(set(data["word"].values))
n_words = len(words)
print ('Number of words:', n_words)

# calculate tags
tags = list(set(data["tag"].values))
n_tags = len(tags)
print ('Number of tags:', n_tags)
print ('Type of tags:', tags)

# create output folder for x and y
gfile.MakeDirs(os.path.dirname(args.output_x_path))
gfile.MakeDirs(os.path.dirname(args.output_y_path))

# preprocess text
processor = TextPreprocessor(140)
processor.fit(sentences_list)
processor.labels = list(set(data["tag"].values))

X = processor.transform(sentences_list)

# preprocess tags
tag2idx = {t: i for i, t in enumerate(tags)}
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

with gfile.GFile(args.output_x_path, 'w') as output_X:
    pickle.dump(X, output_X)

with gfile.GFile(args.output_y_path, 'w') as output_y:
    pickle.dump(y, output_y)

# writing x and y path to a file for downstream tasks
Path(args.output_x_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_x_path_file).write_text(args.output_x_path)

Path(args.output_y_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_y_path_file).write_text(args.output_y_path)
