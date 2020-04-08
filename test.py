import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.train import AdamOptimizer as Adam
import numpy as np
import matplotlib
import cv2 as cv
matplotlib.use("Agg")
import os 
import time
import matplotlib.pyplot as plt

TRAINING_PATH = "../training_data_lab_128_128.npy"
DATA_PATH = "./"

MODEL_PATH = os.path.join(DATA_PATH,"Models")





MODEL_PATH = os.path.join(DATA_PATH,"Models")
GENERATOR_PATH = os.path.join(MODEL_PATH,"color_generator_80.h5")
# DISCRIMINATOR_PATH = os.path.join(MODEL_PATH,"color_discriminator_main.h5")

EPOCHS = 50
BATCH_SIZE = 32
BUFFER_SIZE = 60000




def getDataset(path):
	if not os.path.isfile(TRAINING_PATH):
		print("No pickle found... Please run with the correct file")
		return None
	else:
		print("Loading previous training pickle...")
		training_data = np.load(TRAINING_PATH)
		training_data = training_data.astype(np.float32)
		training_data = ((training_data/127.5)-1)
		return tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


train_dataset = getDataset(TRAINING_PATH)



def save_images(cnt,dataset):
	sample_images = tf.convert_to_tensor([i.numpy() for i in dataset.take(1)])
	last_dimension_axis = len(sample_images.shape) - 1
	y, u, v = tf.split(sample_images, 3, axis=last_dimension_axis)
	generated_images = generator.predict(y[0])
	
	
	generated_images = tf.concat([y[0], generated_images],3) 
	generated_images = tf.multiply(tf.add(generated_images, 1), 127.5)
	sample_images = tf.multiply(tf.add(sample_images, 1), 127.5)
	generated_images = generated_images.numpy()
	sample_images = sample_images.numpy()
	generated_images = [cv.cvtColor(i, cv.COLOR_LAB2RGB) for i in generated_images.astype(np.uint8)]
	sample_images = [cv.cvtColor(i, cv.COLOR_LAB2RGB) for i in sample_images[0].astype(np.uint8)]
	

    

	fig = plt.figure(figsize=(20, 10))

	for i in range(16):
		plt.subplot(4,8,(2*i) +1)
		plt.xticks([])
		plt.yticks([])
		plt.title("Ground Truth")
		plt.imshow(sample_images[i])
		plt.subplot(4,8,(2*i) + 2)
		plt.xticks([])
		plt.yticks([])
		plt.title("Model Generated")
		plt.imshow(generated_images[i])
	fig.savefig(os.path.join(DATA_PATH,f'output/test_{cnt}.png'), dpi =fig.dpi)
	plt.close(fig)
	print(f"Saved Image: test_{cnt}.png")


print("Loading model from memory")
if os.path.isfile(GENERATOR_PATH):
	generator = tf.keras.models.load_model(GENERATOR_PATH)
	print("Generator loaded")
else:

	print("No generator file found")


save_images(1000, train_dataset)
