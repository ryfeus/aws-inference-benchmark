import os
from time import perf_counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
COMPILED_MODEL_DIR = './inception_v3_neuron/'
predictor_inferentia = tf.contrib.predictor.from_saved_model(COMPILED_MODEL_DIR)

n = 10000
list_timings = []
for i in range(n):
	t0 = perf_counter()
	infa_rslts = predictor_inferentia({'input': np.random.rand(1,299,299,3)});
	list_timings.append(perf_counter() - t0)

print("Mean inference", np.mean(list_timings))
print("Std inference", np.std(list_timings))

t0 = perf_counter()
for i in range(n):
	infa_rslts = predictor_inferentia({'input': np.random.rand(1,299,299,3)});

print("Mean inference", (perf_counter() - t0)/n)
