import argparse
import numpy as np

from keras.callbacks import TensorBoard

from tensorflow.python.lib.io import file_io
from sklearn.model_selection import train_test_split

import pandas as pd
from pandas.compat import StringIO

import os
#import hypertune

import trainer.model as model
import trainer.preprocessor as preprocessor


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data')

  parser.add_argument(
      '--epochs',
      help='Epochs to train')

  parser.add_argument(
      '--dropout',
      default=0.1,
      type=float,
      help='Dropout')
  
  parser.add_argument(
      '--lstmunits',
      default=100,
      type=int,
      help='LSTM Units')

  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='local or GCS location for writing checkpoints and exporting models')

  args, _ = parser.parse_known_args()
  return args

def read_data(gcs_path):
   print('downloading csv file from', gcs_path)     
   file_stream = file_io.FileIO(gcs_path, mode='r')
   data = pd.read_csv(StringIO(file_stream.read()),encoding="utf-8", error_bad_lines=False)
   data = data.fillna(method="ffill").head(2000)

   data=data.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos", "shape"],axis=1)

   data.head(10)

   return data

def train(args, preprocessed_data):
    n_words, n_tags, X, y = preprocessed_data

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create model
    bi_lstm_model = model.create_keras_model(n_words=n_words, 
                                             n_tags=n_tags, 
                                             units=args.lstmunits,
                                             maxlen=140, 
                                             dropout=args.dropout)
    # tensorboard
    tensorboard = TensorBoard(
      log_dir=os.path.join(args.job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

    # callbacks
    callbacks = [tensorboard]

    # train bi lstm model
    history = bi_lstm_model.fit(
        X_train, 
        np.array(y_train), 
        batch_size=1024, 
        epochs=3, 
        validation_split=0.1, 
        verbose=1,
        callbacks=callbacks)

    # evaluate
    score = bi_lstm_model.evaluate(X_test, np.array(y_test), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # hyperparameter
    '''
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=score[1]
        )
    '''
    
    ## save model
    model.save_model(bi_lstm_model, args.job_dir)

    return

if __name__ == '__main__':
    args = get_args()

    data = read_data(args.train_file)
    preprocessed_data = preprocessor.preprocess(data)
    train(args, preprocessed_data)
