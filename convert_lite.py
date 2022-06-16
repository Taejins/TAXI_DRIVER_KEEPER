import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.models import load_model

converter = lite.TFLiteConverter.from_saved_model('./mod2')
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
# converter.allow_custom_ops = True
tfmodel = converter.convert()
open ("LSTM_model.tflite" , "wb") .write(tfmodel)