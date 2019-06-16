'''
np.save('x_train.npy', X_train)
np.save('y_train.npy', y_train)

np.save('x_test.npy', X_test)
np.save('y_test.npy', y_test)

X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
X_train


with tf.gfile.GFile('test.npy', 'wb') as file:
  file.write(X_train.tobytes())
  
#test = tf.gfile.GFile('test.npy', "rb").read()

with tf.gfile.Open("test.npy", "rb") as file_:
  info = np.load(file_)
  
#est = np.load(tf.gfile.GFile('test.npy', 'rb').read())
#np.load(test)

information = { "a": 1, "b": 2 }
np.save("a.npy", information)

with tf.gfile.Open("a.npy", "rb") as file_:
  info = np.load(file_, allow_pickle=True)

  if type(info) is np.ndarray:
  print('jeah')

X_train
'''

import argparse
import os
from pathlib import Path
from tensorflow import gfile
import numpy as np
import pickle  

def load_feature(input_x_path):
  with gfile.Open(input_x_path, 'rb') as input_x_file:
    return pickle.loads(input_x_file.read())

def load_label(input_y_path):
  with gfile.Open(input_y_path, 'rb') as input_y_file:
    return pickle.loads(input_y_file.read())

# Defining and parsing the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input-x-path', type=str, help='')
parser.add_argument('--input-y-path', type=str, help='')

parser.add_argument('--param1', type=int, default=100, help='Parameter 1.')

parser.add_argument('--output-x-path', type=str, help='')
parser.add_argument('--output-x-path-file', type=str, help='')

parser.add_argument('--output-y-path', type=str, help='')
parser.add_argument('--output-y-path-file', type=str, help='')

args = parser.parse_args()

print(args.input_x_path)
print(args.input_y_path)

X_train = load_feature(args.input_x_path)
y_train = load_label(args.input_y_path)

#print(X_train)
#print(y_train)

# writing x and y path to a file for downstream tasks
Path(args.output_x_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_x_path_file).write_text('test')

Path(args.output_y_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_y_path_file).write_text('test')
