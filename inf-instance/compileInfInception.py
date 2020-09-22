import os
import time
import shutil
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create a workspace
WORKSPACE = './ws_inception_v3'
os.makedirs(WORKSPACE, exist_ok=True)

# Prepare export directory (old one removed)
model_dir = os.path.join(WORKSPACE, 'inception_v3')
compiled_model_dir = os.path.join(WORKSPACE, 'inception_v3_neuron')
shutil.rmtree(model_dir, ignore_errors=True)
shutil.rmtree(compiled_model_dir, ignore_errors=True)

# Instantiate Keras InceptionV3 model
keras.backend.set_learning_phase(0)
keras.backend.set_image_data_format('channels_last')

model = InceptionV3(weights='imagenet')

# Export SavedModel
tf.saved_model.simple_save(
    session            = keras.backend.get_session(),
    export_dir         = model_dir,
    inputs             = {'input': model.inputs[0]},
    outputs            = {'output': model.outputs[0]})

# Compile using Neuron
tfn.saved_model.compile(model_dir, compiled_model_dir)    

# Prepare SavedModel for uploading to Inf1 instance
shutil.make_archive('./inception_v3_neuron', 'zip', WORKSPACE, 'inception_v3_neuron')