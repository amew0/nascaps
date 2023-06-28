# this is a draft of NSGA2 algorith
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf 
import numpy as np
from keras import optimizers

from PIL import Image
import time
import random
import hw_estimator
import svhn
from wrapper import CapsNet
import json, gzip
import shutil
from random import randint
import random
from math import ceil
from keras import backend as K
from keras import utils, callbacks
#from keras.models import Model
#from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.layer_utils import count_params
from copy import deepcopy
from paretoarchive import PyBspTreeArchive
import uuid
from timeout_callback import TimeoutCallback
import sys
sys.path.append("..")

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2 #config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

gene = [[0, 28, 1, 1, 9, 1, 28, 8, 1], 
        [0, 28, 8, 1, 5, 1, 28, 4, 1], 
        [0, 28, 4, 1, 3, 1, 28, 4, 1], 
        [2, 28, 4, 1, 3, 2, 14, 32, 4], 
        [2, 14, 32, 4, 3, 2, 7, 32, 8], 
        [2, 7, 32, 8, 3, 2, 4, 32, 8], 
        [2, 4, 32, 8, 4, 1, 4, 10, 8], 
        [-1], 
        [1]]

input_shape = (56,56,1)
n_class = 10
routings = 3

mt, me, mm = CapsNet(gene=gene,
        input_shape=input_shape,
        n_class = n_class,
        routings=routings)

# print()
# import visualkeras
# # from PIL import ImageFont
# # font=ImageFont.truetype("arial.ttf",12)
# visualkeras.layered_view(mt,
#                 legend=True,
#                 spacing=30,
#                 # font=font
#                 )

import tensorflow as tf
import numpy as np

# Set configuration for GPU memory usage
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
set_session(sess)

# Load the MNIST dataset
with np.load('nsga/data/mnist.npz') as data:
    x_train = data['x_train']
    y_train = data['y_train']

from main import resize
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
y_train = utils.to_categorical(y_train.astype('float32'))

desired_size = 56
x_train = resize(x_train,desired_size)
# Define the batch size and total number of samples
batch_size = 64
num_samples = x_train.shape[0]

# Create a TensorFlow Dataset from the data
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(num_samples)
dataset = dataset.batch(batch_size)

iterator = dataset.make_one_shot_iterator()
data = iterator.get_next()

gene = [[0, 28, 1, 1, 9, 1, 28, 8, 1], 
        [0, 28, 8, 1, 5, 1, 28, 4, 1], 
        [0, 28, 4, 1, 3, 1, 28, 4, 1], 
        [2, 28, 4, 1, 3, 2, 14, 32, 4], 
        [2, 14, 32, 4, 3, 2, 7, 32, 8], 
        [2, 7, 32, 8, 3, 2, 4, 32, 8], 
        [2, 4, 32, 8, 4, 1, 4, 10, 8], 
        [-1], 
        [1]]

input_shape = (56,56,1)
n_class = 10
routings = 3

from keras import models
model:models.Model = models.Model()
model, _, _ = CapsNet(gene=gene,
        input_shape=input_shape,
        n_class = n_class,
        routings=routings) 


from main import margin_loss
lam_recon = 0.392
lr = 0.001
model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., lam_recon],
                  metrics={'capsnet': 'accuracy'})

epochs = 5
# Perform the training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    while True:
        try:
            batch_data = sess.run(data)
            loss, metrics_values = model.train_on_batch([batch_data[0], batch_data[1]],
                                                        [batch_data[1], batch_data[0]])
            print(f"Loss: {loss}, Metrics: {metrics_values}")
        except tf.errors.OutOfRangeError:
            break
model.fit_generator
model.fit([batch_data[0], ], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])