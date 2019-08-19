from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import VGG16
from datasetopeation.jadeClassifyTFRecords import LoadClassifyTFRecord
model = VGG16()


def loadData(train_path,test_path,batch_size):
  train_ds = LoadClassifyTFRecord(train_path, batch_size, shuffle=True, repeat=False, is_train=True)
  test_ds = LoadClassifyTFRecord(test_path,batch_size, shuffle=True, repeat=False, is_train=False)
  return train_ds, test_ds

train_ds,test_ds = loadData("/home/jade/Data/sdfgoods/TFRecords/sdfgoods_train.tfrecord","/home/jade/Data/sdfgoods/TFRecords/sdfgoods_test.tfrecord",32)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()




optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))