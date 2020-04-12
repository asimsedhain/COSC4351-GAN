import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2 as cv

# Helper function that imports and returns tensorflow dataset
def get_dataset(path, buffer_size, batch_size):
	if not os.path.isfile(path):
		print("No pickle found... Please run with the correct file")
		return None
	else:
		print("Loading previous training pickle...")
		training_data = np.load(path)
		training_data = training_data.astype(np.float32)
		training_data = ((training_data/127.5)-1)
		return tf.data.Dataset.from_tensor_slices(training_data).shuffle(buffer_size).batch(batch_size)


# Helper method for saving an output file while training.
def save_images(path, cnt,dataset):
       sample_images = tf.convert_to_tensor([i.numpy() for i in dataset.take(1)])
       last_dimension_axis = len(sample_images.shape) - 1
       y, u, v = tf.split(sample_images, 3, axis=last_dimension_axis)
       generated_images = generator.predict(y[0])
       
       
       generated_images = tf.concat([y[0], generated_images],3) 
       
       
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
       fig.savefig(os.path.join(path,f'test_{cnt}.png'), dpi =fig.dpi)
       plt.close(fig)
       print(f"Saved Image: test_{cnt}.png")