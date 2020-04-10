
# Importing libraries


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
from model import build_discriminator
from model import build_generator
from model import discriminator_loss
from model import generator_loss
import sys
import traceback


tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

mirrored_strategy = tf.distribute.MirroredStrategy()
# configration


"""Change the INITIAL_TRAINING variable to decide if model is to be loaded from memory or trained again."""
INITIAL_TRAINING = False
Checkpoint = 85


# Generation resolution - Must be square 
# Training data is also scaled to this.
# Note GENERATE_RES 4 or higher  will blow Google CoLab's memory and have not
# been tested extensivly.
GENERATE_RES = 4 # Generation resolution factor (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3


BATCH_SIZE_PER_REPLICA = 32
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync



# Configuration
TRAINING_PATH = "../training_data_lab_128_128.npy"
DATA_PATH = "./testing"

MODEL_PATH = os.path.join(DATA_PATH,"Models")





MODEL_PATH = os.path.join(DATA_PATH,"Models")
GENERATOR_PATH = os.path.join(MODEL_PATH,f"color_generator_{checkpoint}.h5")
DISCRIMINATOR_PATH = os.path.join(MODEL_PATH,"color_discriminator_{checkpoint}.h5")

EPOCHS = 50
BATCH_SIZE = 32
BUFFER_SIZE = 60000

print(f"Will generate {GENERATE_SQUARE}px square images.")





# def getDataset(path):
# 	train_path = pathlib.Path(path)
# 	list_ds = tf.data.Dataset.list_files(str(train_path/'*'))
# 	def parse_image(filename):
# 		image = tf.io.read_file(filename)
# 		image = tf.image.decode_jpeg(image, channels = 3, try_recover_truncated= True)
# 		image = tf.image.convert_image_dtype(image, tf.float32)
# 		image = tf.image.rgb_to_yuv(image)
# 		image = tf.image.resize(image, [128, 128])
# 		last_dimension_axis = len(image.shape) - 1
# 		y, u, v = tf.split(image, 3, axis=last_dimension_axis)
# 		y = tf.subtract(y, 0.5)
# 		preprocessed_yuv_images = tf.concat([y, u, v], axis=last_dimension_axis)
# 		return preprocessed_yuv_images
# 	img_ds = list_ds.map(parse_image)
# 	return img_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def getDataset(path):
	if not os.path.isfile(TRAINING_PATH):
		print("No pickle found... Please run with the correct file")
		return None
	else:
		print("Loading previous training pickle...")
		training_data = np.load(TRAINING_PATH)
		training_data = training_data.astype(np.float32)
		training_data = ((training_data/127.5)-1)
		return tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)


train_dataset = getDataset(TRAINING_PATH)
dist_dataset = mirrored_strategy.make_dataset_iterator(train_dataset)



# Helper method for saving an output file while training.
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







# Checks if you want to continue training model from disk or start a new
with mirrored_strategy.scope():
	if(INITIAL_TRAINING):	
		print("Initializing Generator and Discriminator")
		generator = build_generator()
		discriminator = build_discriminator()
		print("Generator and Discriminator initialized")
		generator_optimizer = Adam(2e-4,0.5)
		discriminator_optimizer = Adam(2e-4,0.5)

	else:
		print("Loading model from memory")
		if os.path.isfile(GENERATOR_PATH) :
			generator = tf.keras.models.load_model(GENERATOR_PATH)
			print("Generator loaded")
		else:
		
			print("No generator file found")
		if os.path.isfile(DISCRIMINATOR_PATH):
			
			discriminator = tf.keras.models.load_model(DISCRIMINATOR_PATH)
			print("Discriminator loaded")
		else:
	
			print("No discriminator file found")
		









with mirrored_strategy.scope():
	def train_step(images):
		seed = tf.reshape(images[:,:, :, 0], (images.shape[0], GENERATE_SQUARE, GENERATE_SQUARE, 1))
		real = images[:,:, :, 1:3]

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_images = generator(seed, training=True)
			real_output = discriminator(real, training=True)
			fake_output = discriminator(generated_images, training=True)

			gen_loss = generator_loss(fake_output, real, generated_images)
			disc_loss = discriminator_loss(real_output, fake_output)


			gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
			gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

			generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
			discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
		gen_loss = tf.reduce_sum(gen_loss)*(1./GLOBAL_BATCH_SIZE)
		disc_loss = tf.reduce_sum(disc_loss)*(1./GLOBAL_BATCH_SIZE)
		return gen_loss,disc_loss

	def dist_train_step(images):
		return train_step(images)	
	

def train(dataset, epochs):
	start = time.time()
	
	for epoch in range(epochs):
		epoch_start = time.time()
		dist_dataset.initialize()
		gen_loss, disc_loss = 0.0, 0.0
		num_batches = 0
		with mirrored_strategy.scope():	
			try:
				while True:
					losses = mirrored_strategy.experimental_run(dist_train_step, dist_dataset)		
					losses = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses[0]), mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses[1])
					gen_loss +=losses[0]
					disc_loss+=losses[1]
					num_batches+=1
			except Exception as err:
				print(traceback.format_exc())
				pass
		
			save_images(epoch+checkpoint, train_dataset)
			if(epoch%5==0):
				print(f"Saving Model for epoch {epoch}")
				generator.save(os.path.join(MODEL_PATH,f"color_generator_{epoch+checkpoint}.h5"))
				discriminator.save(os.path.join(MODEL_PATH,f"color_discriminator_{epoch+checkpoint}.h5"))


		epoch_elapsed = time.time()-epoch_start
		
		tf.print(f'Epoch {epoch+1+checkpoint}, gen loss={gen_loss/num_batches},disc loss={disc_loss/num_batches}, Epoch Time:{(epoch_elapsed)}, Num of Batches: {num_batches}', output_stream=sys.stderr)

		
		

	elapsed = time.time()-start
	print (f'Training time: {(elapsed)}')

train(train_dataset, 100)





# saving the model to disk

with mirrored_strategy.scope():
	generator.save(os.path.join(MODEL_PATH,"color_generator_main.h5"))
	discriminator.save(os.path.join(MODEL_PATH,"color_discriminator_main.h5"))
	print(f"Models save to {MODEL_PATH}")

