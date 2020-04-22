import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2 as cv
import pathlib


# Helper function that imports and returns tensorflow dataset
# def get_dataset(path, buffer_size, batch_size):
# 	if not os.path.isfile(path):
# 		print("No pickle found... Please run with the correct file")
# 		return None
# 	else:
# 		print("Loading previous training pickle...")
# 		training_data = np.load(path)
# 		training_data = training_data.astype(np.float32)
# 		training_data = ((training_data/127.5)-1)
# 		return tf.data.Dataset.from_tensor_slices(training_data).shuffle(buffer_size).batch(batch_size)


# # Helper method for saving an output file while training.
# def save_images(path, cnt,dataset):
#        sample_images = tf.convert_to_tensor([i.numpy() for i in dataset.take(1)])
#        last_dimension_axis = len(sample_images.shape) - 1
#        y, u, v = tf.split(sample_images, 3, axis=last_dimension_axis)
#        generated_images = generator.predict(y[0])
       
       
#        generated_images = tf.concat([y[0], generated_images],3) 
       
       
#        generated_images = tf.concat([y[0], generated_images],3) 
#        generated_images = tf.multiply(tf.add(generated_images, 1), 127.5)
#        sample_images = tf.multiply(tf.add(sample_images, 1), 127.5)
#        generated_images = generated_images.numpy()
#        sample_images = sample_images.numpy()
#        generated_images = [cv.cvtColor(i, cv.COLOR_LAB2RGB) for i in generated_images.astype(np.uint8)]
#        sample_images = [cv.cvtColor(i, cv.COLOR_LAB2RGB) for i in sample_images[0].astype(np.uint8)]
       

#        fig = plt.figure(figsize=(20, 10))

#        for i in range(16):
#                plt.subplot(4,8,(2*i) +1)
#                plt.xticks([])
#                plt.yticks([])
#                plt.title("Ground Truth")
#                plt.imshow(sample_images[i])
#                plt.subplot(4,8,(2*i) + 2)
#                plt.xticks([])
#                plt.yticks([])
#                plt.title("Model Generated")
#                plt.imshow(generated_images[i])
#        fig.savefig(os.path.join(path,f'test_{cnt}.png'), dpi =fig.dpi)
#        plt.close(fig)
#        print(f"Saved Image: test_{cnt}.png")

def get_dataset(path, buffer_size, batch_size, num_workers, worker_index):
	train_path = pathlib.Path(path)
	list_ds = tf.data.Dataset.list_files(str(train_path/'*'))
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
	# img_ds = list_ds.map(parse_image)
	
	img_ds = list_ds.shard(num_workers, worker_index).map(parse_image).repeat(-1).shuffle(buffer_size).batch(batch_size)
	return img_ds

# Helper method for saving an output file while training.
def save_images(path,cnt,dataset, generator, save_or_not):
	sample_images = tf.convert_to_tensor([i.numpy() for i in dataset.take(1)])
	#sample_images_input = tf.reshape(sample_images[0,0:16, :, :, 0],(16, 128, 128, 1))
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
	
	if(save_or_not):
		fig.savefig(os.path.join(path,f'test_{cnt}.png'), dpi =fig.dpi)
	plt.close(fig)
	print(f"Saved Image: test_{cnt}.png")
