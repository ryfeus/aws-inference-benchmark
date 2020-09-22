import boto3
import numpy as np
import tensorflow as tf
import os.path
import re
from urllib.request import urlretrieve
import json
from io import BytesIO
from time import perf_counter

SESSION = None
strBucket = 'serverlessdeeplearning'

def handler(event, context):
    global strBucket
    if not os.path.exists('/tmp/imagenet/'):
        os.makedirs('/tmp/imagenet/')

    strFile = '/tmp/imagenet/inputimage.png'

    downloadFromS3(strBucket,'imagenet/inputimage.png',strFile)

    global SESSION
    if SESSION is None:
        SESSION = tf.InteractiveSession()
        create_graph()

    softmax_tensor = tf.get_default_graph().get_tensor_by_name('softmax:0')
    num_of_cycles = event.get('num_of_cycles', 10)
    batch_size = event.get('batch_size', 1)

    list_imgs = []
    list_timings = []
    image_data = tf.gfile.FastGFile(strFile, 'rb').read()

    print('Started prediction')

    for i in range(num_of_cycles):
        t0 = perf_counter()
        predictions = SESSION.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        list_timings.append(perf_counter() - t0)

    print("Mean inference", np.mean(list_timings))
    print("Std inference", np.std(list_timings))
    return {'Mean inference': np.mean(list_timings), 'Std inference': np.std(list_timings)}

def downloadFromS3(strBucket,strKey,strFile):
    s3_client = boto3.client('s3')
    s3_client.download_file(strBucket, strKey, strFile)

def getObject(strBucket,strKey):
    s3_client = boto3.client('s3')
    s3_response_object = s3_client.get_object(Bucket=strBucket, Key=strKey)
    return s3_response_object['Body'].read()  

def getObjectParallel(strBucket,strKey):
    s3_client = boto3.client('s3')
    with BytesIO() as data:
        s3_client.download_fileobj(strBucket, strKey, data)
        return data.getvalue()

def create_graph():
    global strBucket
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(getObjectParallel(strBucket,'imagenet/classify_image_graph_def.pb'))
    _ = tf.import_graph_def(graph_def, name='')
