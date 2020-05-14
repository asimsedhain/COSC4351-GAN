import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime

# Parses the image for the given filename
def parse_image(filename):
	image = tf.io.read_file(filename)
	image = tf.image.decode_jpeg(image, channels = 3, try_recover_truncated= True)
	image = tf.image.convert_image_dtype(image, tf.float32)
	image = tf.image.rgb_to_yuv(image)
	image = tf.image.resize(image, [128, 128])
	last_dimension_axis = len(image.shape) - 1
	y, u, v = tf.split(image, 3, axis=last_dimension_axis)
	y = tf.subtract(y, 0.5)
	preprocessed_yuv_images = tf.concat([y, u, v], axis=last_dimension_axis)
	return preprocessed_yuv_images

# Returns the dataset from the images from the path varibale
def get_dataset(path, buffer_size, batch_size):
	train_path = pathlib.Path(path)
	list_ds = tf.data.Dataset.list_files(str(train_path/'*'))
	
	img_ds = list_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size)).map(parse_image).batch(batch_size)
	return img_ds

def get_sample(path):
	train_path = pathlib.Path(path)
	list_ds = tf.data.Dataset.list_files(str(train_path/'*'))
	list_ds = list_ds.take(16)
	img_ds = list_ds.map(parse_image).batch(16)
	return img_ds

# Helper method for saving an output file while training.
def save_images(path,cnt,sample_images, generator):
	
	last_dimension_axis = len(sample_images.shape) - 1
	y, u, v = tf.split(sample_images, 3, axis=last_dimension_axis)
	
	generated_images = generator.predict(y[0].numpy())
	y = tf.add(y, 0.5)
	#sample_images_input = tf.add(sample_images_input, 0.5)
	
	generated_images = tf.concat([y[0], generated_images],3) 

	generated_images = tf.image.yuv_to_rgb(generated_images)
	generated_images = generated_images.numpy()
	sample_images = tf.image.yuv_to_rgb(tf.concat([y, u, v], axis=last_dimension_axis))
	sample_images = sample_images.numpy()
    

	fig = plt.figure(figsize=(20, 10))

	for i in range(16):
		plt.subplot(4,8,(2*i) +1)
		plt.xticks([])
		plt.yticks([])
		plt.title("Ground Truth")
		plt.imshow(sample_images[0, i])
		plt.subplot(4,8,(2*i) + 2)
		plt.xticks([])
		plt.yticks([])
		plt.title("Model Generated")
		plt.imshow(generated_images[i])
	
	
	fig.savefig(os.path.join(path,f'test_{cnt}.png'), dpi =fig.dpi)
	plt.close(fig)
	print(f"Saved Image: test_{cnt}.png")

# Logger object for loggint from the correct rank
# Logs with a timestamp and to the passed output stream
class logger(object):

	
	def print(self, str, output_stream):
		tf.print(f"{datetime.now()}: {str}", output_stream = output_stream)
