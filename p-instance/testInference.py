# sudo pip3 install tensorflow-gpu


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from time import perf_counter

model = InceptionV3(weights='imagenet', include_top=True)
model.predict(np.random.rand(1,299,299,3))

n = 100
batch_size = 1
list_imgs = []
list_timings = []

for i in range(n):
	list_imgs.append(np.random.rand(batch_size,299,299,3))

for i in range(n):
	t0 = perf_counter()
	model.predict(list_imgs[i])
	list_timings.append(perf_counter() - t0)

print("Mean inference", np.mean(list_timings))
print("Std inference", np.std(list_timings))

t0 = perf_counter()
for i in range(n):
	model.predict(list_imgs[i])

print("Mean inference", (perf_counter() - t0)/n)