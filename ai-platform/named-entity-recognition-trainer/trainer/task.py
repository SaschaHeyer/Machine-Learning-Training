import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from pandas.compat import StringIO
import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense
from keras.layers import TimeDistributed, Dropout, Bidirectional

from keras import backend as K


from keras.callbacks import TensorBoard

import os
import tempfile
from datetime import datetime
import hypertune


class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                    s["tag"].values.tolist())]
        
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text

class TextPreprocessor(object):
  
  def __init__(self, max_sequence_length):
    self._max_sequence_length = max_sequence_length
    self._labels = None
    self.number_words = None
    self._tokenizer = None
    
  def fit(self, instances):
    tokenizer = text.Tokenizer(lower=False, filters=[], oov_token=None)
    tokenizer.fit_on_texts(instances)
    self._tokenizer = tokenizer
    self.number_words = len(tokenizer.word_index)
    print(self.number_words)
    
  def transform(self,instances):
    sequences = self._tokenizer.texts_to_sequences(instances)
    padded_sequences = pad_sequences(maxlen=140, sequences=sequences, padding="post",value=self.number_words - 1)
    return padded_sequences

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



def train(args, data):
    getter = SentenceGetter(data)
    sentences = getter.sentences
    sentences_list = [" ".join([s[0] for s in sent]) for sent in sentences]
    sentences_list[0]

    maxlen = max([len(s) for s in sentences])
    print ('Maximum sequence length:', maxlen)

    words = list(set(data["word"].values))
    #words.append("ENDPAD")
    n_words = len(words)
    print ('Number of words:', n_words)

    tags = list(set(data["tag"].values))
    n_tags = len(tags)
    print ('Number of tags:', n_tags)
    print ('Type of tags:', tags)

    processor = TextPreprocessor(140)
    processor.fit(sentences_list)
    processor.labels = list(set(data["tag"].values))

    X = processor.transform(sentences_list)

    tag2idx = {t: i for i, t in enumerate(tags)}

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])

    from keras.utils import to_categorical
    y = [to_categorical(i, num_classes=n_tags) for i in y]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from keras.models import Model, Input
    from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

    #logs_path = args.job_dir + '/logs/' + datetime.now().isoformat()
    #print('Using logs_path located at {}'.format(logs_path))

    input = Input(shape=(140,))
    model = Embedding(input_dim=n_words, output_dim=140, input_length=140)(input)
    model = Dropout(args.dropout)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer

    model = Model(input, out)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




    model.summary()

    tensorboard = TensorBoard(
      log_dir=os.path.join(args.job_dir, 'logs'),
      histogram_freq=0,
      write_graph=True,
      embeddings_freq=0)

    callbacks = [tensorboard]

    history = model.fit(
        X_train, 
        np.array(y_train), 
        batch_size=1024, 
        epochs=3, 
        validation_split=0.1, 
        verbose=1,
        callbacks=callbacks)



    score = model.evaluate(X_test, np.array(y_test), verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=score[1]
        )
    
    print('save model')
    model.save('keras_saved_model.h5')
    #print('write model to folder')
    with file_io.FileIO('keras_saved_model.h5', mode='rb') as input_f:
        with file_io.FileIO(args.job_dir + '/keras_saved_model.h5', mode='wb+') as output_f:
            output_f.write(input_f.read())

    return

if __name__ == '__main__':
    args = get_args()

    data = read_data(args.train_file)
    train(args, data)
