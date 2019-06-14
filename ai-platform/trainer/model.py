
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

import tensorflow as tf

#maxlen = 140
#dropout= 0.1
def create_keras_model(n_words, n_tags, units, maxlen, dropout):

  input = Input(shape=(maxlen,))
  model = Embedding(input_dim=n_words, output_dim=maxlen, input_length=maxlen)(input)
  model = Dropout(dropout)(model)
  model = Bidirectional(LSTM(units=units, return_sequences=True, recurrent_dropout=0.1))(model)
  out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

  model = Model(input, out)

  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

  model.summary()
  return model
