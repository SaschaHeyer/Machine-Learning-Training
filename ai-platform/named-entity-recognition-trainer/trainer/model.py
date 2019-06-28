from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

from tensorflow.python.lib.io import file_io

def create_keras_model(n_words, 
                       n_tags, 
                       units, 
                       maxlen, 
                       dropout):

  input = Input(shape=(maxlen,))
  model = Embedding(input_dim=n_words, output_dim=maxlen, input_length=maxlen)(input)
  model = Dropout(dropout)(model)
  model = Bidirectional(LSTM(units=units, return_sequences=True, recurrent_dropout=0.1))(model)
  out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

  model = Model(input, out)

  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

  model.summary()
  return model

def save_model(model, job_dir):
  model.save('keras_saved_model.h5')
  
  print('write model to folder', job_dir)
  with file_io.FileIO('keras_saved_model.h5', mode='rb') as input_f:
    with file_io.FileIO(job_dir + '/' + 'keras_saved_model.h5', mode='wb+') as output_f:
      output_f.write(input_f.read())
        
  return