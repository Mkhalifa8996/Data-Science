import tensorflow as tf
import tf_keras  # Force TF-Keras (Keras 2 compatible) to prevent Keras 3 compatibility errors
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile,random, os, datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

def extract_files(source_dir, target_dir):
  zip_ref = zipfile.ZipFile(from_dir)
  zip_ref.extractall(to_dir)
  zip_ref.close()

def walk_through_dir(dir_path)
for dirpath, dirnames, filenames in os.walk(dir_path):
  print(f"There are {len(dirnames)} directories  and {len(filenames)} files in {dirpath}")

def view_history(history):
  epochs = tf.range(len(history.history["loss"]))
  # Plot the loss
  plt.figure(figsize=(7,5))
  plt.plot(epochs,history.history["loss"])
  plt.plot(epochs,history.history["val_loss"])
  # Plot the accuracy
  plt.figure(figsize=(7,5))
  plt.plot(epochs,history.history["accuracy"])
  plt.plot(epochs,history.history["val_accuracy"])

"""
Setting up callbacks
callbacks are extra functionality
you can add to your models to be performed during or after training**

1.   TensorBoard callback: track experiments
2.   ModelCheckpoint callback: model check point
1.   EarlyStopping callback: stop a model from training(before it trains too long and overfit
"""
# Create TensorBoard callback
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf_keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving Tensorboard log file to: {log_dir}")
  return tensorboard_callback

"""
Creating models with Tensorflow Hub
Tensorflow Hub is a place to find pretrained machine learning models(ready to be applied and fine tuned for a specific problem)
We will use the following feature vector model: https://www.kaggle.com/models/tensorflow/efficientnet
"""
def create_model(model_url, num_classes=10):
  """
  Take a TensorFlow Hub URL and create a Keras Sequential model with it.
  Args:
    model_url (str): A tensorFlow Hub extraction URL,
    num_classes (int): Number of output neurons in the output layer,
    should be equal to number of target classes, default 10.
    
  Returns:
    An uncompiled Keras Sequential model with model_url as feature
     extractor layer and Dense output layer with num_classes output neurons.
  """
  feature_extractor_layer = hub.KerasLayer(
      model_url,
      trainable=False,
      name="feature_extraction_layer",
      input_shape=IMAGE_SHAPE+(3,)
  )

  model = keras.Sequential([
      feature_extractor_layer,
      keras.layers.Dense(num_classes, activation="softmax", name="output_layer")
  ])

  return model
