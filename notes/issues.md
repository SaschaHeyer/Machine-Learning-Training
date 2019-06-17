

## Keras
### ('Expected `model` argument to be a `Model` instance, got ', <keras.engine.training.Model object at 0x7f3ecf85b908>)

Change ```from keras.models import Model``` to ```from tensorflow.keras.models import Model```

tf.keras or Keras?

Keras is now part of TensorFlow, just use tf.keras. 

Similar Issues
* https://github.com/keras-team/keras/issues/9310