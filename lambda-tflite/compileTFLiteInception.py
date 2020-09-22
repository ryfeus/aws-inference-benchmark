import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from time import process_time

model = InceptionV3(weights='imagenet')
model.save("InceptionV3.h5")

converter = tf.lite.TFLiteConverter.from_keras_model_file('InceptionV3.h5',
	input_shapes={'input_1' : [1,299,299,3]}
	)
tflite_model = converter.convert()
open("InceptionV3.tflite", "wb").write(tflite_model)