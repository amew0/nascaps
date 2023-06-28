import tensorflow as tf
import numpy as np

with np.load('mnist.npz') as data:
    x_train = data['x_train']
    y_train = data['y_train']

batch_size = 64
num_samples = x_train.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(num_samples)
dataset = dataset.batch(batch_size)

iterator = dataset.make_one_shot_iterator()
data = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            batch_data = sess.run(data)
            print(batch_data[0].shape, batch_data[1].shape)
        except tf.errors.OutOfRangeError:
            break
