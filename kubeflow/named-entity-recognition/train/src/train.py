import argparse
from pathlib import Path
from tensorflow import gfile
import numpy as np
import pickle  
from tensorflow.python.lib.io import file_io
import json
import os

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

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
parser.add_argument('--input-job-dir', type=str, help='')

parser.add_argument('--input-tags', type=int, help='')
parser.add_argument('--input-words', type=int, help='')

parser.add_argument('--output-model-path', type=str, help='')
parser.add_argument('--output-model-path-file', type=str, help='')

args = parser.parse_args()

print(args.input_x_path)
print(args.input_y_path)
print(args.input_tags)
print(args.input_words)

X = load_feature(args.input_x_path)
y = load_label(args.input_y_path)

#print(X_train)
#print(y_train)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# initialize tensorboard
tensorboard = TensorBoard(
      log_dir=os.path.join(args.input_job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

callbacks = [tensorboard]

# model
input = Input(shape=(140,))
model = Embedding(input_dim=args.input_words, output_dim=140, input_length=140)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(args.input_tags, activation="softmax"))(model)  # softmax output layer
model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=1, validation_split=0.1, verbose=1, callbacks=callbacks)

loss, accuracy = model.evaluate(X_test, np.array(y_test))

metrics = {
        'metrics': [{
          'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
          'numberValue':  accuracy, # The value of the metric. Must be a numeric value.
          'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
        }]
      }

metadata = {
    'outputs' : [{
      'type': 'tensorboard',
      'source': args.input_job_dir,
    }]
}

with open('/mlpipeline-ui-metadata.json', 'w') as f:
  json.dump(metadata, f)

with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
  json.dump(metrics, f)

# writing x and y path to a file for downstream tasks
Path(args.output_x_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_x_path_file).write_text('test')

Path(args.output_y_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_y_path_file).write_text('test')

Path(args.output_model_path_file).parent.mkdir(parents=True, exist_ok=True)
Path(args.output_model_path_file).write_text(args.output_model_path)