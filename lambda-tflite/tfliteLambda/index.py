import boto3
import json
import io
import tflite_runtime.interpreter as tflite
import numpy as np
from time import perf_counter
interpreter = None

def handler(event, context):
  global interpreter
  if interpreter is None:
    interpreter = tflite.Interpreter(model_path="models/InceptionV3.tflite")
    interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  input_shape = input_details[0]['shape']

  list_imgs = []
  list_timings = []
  num_of_cycles = event.get('num_of_cycles', 10)

  for i in range(num_of_cycles):
    list_imgs.append(np.array(np.random.random_sample(input_shape), dtype=np.float32))

  for i in range(num_of_cycles):
    t0 = perf_counter()

    interpreter.set_tensor(input_details[0]['index'], list_imgs[i])
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    list_timings.append(perf_counter() - t0)

  print("Mean inference", np.mean(list_timings))
  print("Std inference", np.std(list_timings))

  return {'Mean inference': np.mean(list_timings), 'Std inference': np.std(list_timings)}

