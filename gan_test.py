
# Importing libraries

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.train import AdamOptimizer as Adam
import numpy as np
import os 
import time
import sys
import psutil


# Need this for distributed training
import horovod.tensorflow as hvd


import matplotlib




# Helper libraries
from models import build_discriminator
from models import build_generator
from models import discriminator_loss
from models import generator_loss
from utils import get_dataset
from utils import get_sample
from utils import save_images
from utils import logger



# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())


tf.enable_eager_execution(config=config)

# logger object
logger = logger(hvd)

# Configration

# Need to use "Agg" for machines without a display. Or it wil result in segmentation fault
matplotlib.use("Agg")

# Training data directory
TRAINING_DATA_PATH = "../test"
DATASET_SIZE = 1000

# All the output and models will be saved inside the checkpoint path
CHECKPOINT_PATH = "./test_horovod"

# Sample images will be stored in the output path
OUTPUT_PATH = os.path.join(CHECKPOINT_PATH, "output") 
# Path for the model. It is inside the checkpoint directory
MODEL_PATH = os.path.join(CHECKPOINT_PATH,"Models")

# If INITIAL_TRAINING is set to False, generator and discriminator will be loaded from the following path
GENERATOR_PATH_PRE = os.path.join(MODEL_PATH,"color_generator_main.h5")
DISCRIMINATOR_PATH_PRE = os.path.join(MODEL_PATH,"color_discriminator_main.h5")

# Path for the final models to be saved to after training
GENERATOR_PATH_FINAL = os.path.join(MODEL_PATH,"color_generator_final.h5")
DISCRIMINATOR_PATH_FINAL = os.path.join(MODEL_PATH,"color_discriminator_final.h5")


# Change the INITIAL_TRAINING variable to decide if model is to be loaded from memory or trained again.
INITIAL_TRAINING = True

# Size of the image. The input data will also be scaled to this amount.
GENERATE_SQUARE = 128

EPOCHS = 200
BATCH_SIZE = 32
BUFFER_SIZE = 2**15



logger.print(f"Will generate {GENERATE_SQUARE}px square images.", output_stream=sys.stdout)



logger.print(f"Images being loaded from {TRAINING_DATA_PATH}", output_stream=sys.stdout)

train_dataset = get_dataset(TRAINING_DATA_PATH, BUFFER_SIZE, BATCH_SIZE)

logger.print(f"Images loaded from {TRAINING_DATA_PATH}", output_stream=sys.stdout)

sample_images = get_sample(TRAINING_DATA_PATH)

sample_images = tf.convert_to_tensor([i.numpy() for i in sample_images.take(1)])


# Checks if you want to continue training model from disk or start a new
# Set INITIAL_TRAINING to true if you want to continue training
if(INITIAL_TRAINING):
	logger.print("Initializing Generator and Discriminator", output_stream=sys.stdout)
	generator = build_generator(image_shape=(GENERATE_SQUARE, GENERATE_SQUARE, 1))
	discriminator = build_discriminator(image_shape=(GENERATE_SQUARE, GENERATE_SQUARE, 2))
	logger.print("Generator and Discriminator initialized", output_stream=sys.stdout)
else:
	logger.print("Loading model from memory", output_stream=sys.stdout)
	if os.path.isfile(GENERATOR_PATH_PRE):
		generator = tf.keras.models.load_model(GENERATOR_PATH_PRE)
		logger.print("Generator loaded", output_stream=sys.stdout)
	else:
		logger.print("No generator file found", output_stream=sys.stdout)
	if os.path.isfile(DISCRIMINATOR_PATH_PRE):
		
		discriminator = tf.keras.models.load_model(DISCRIMINATOR_PATH_PRE)
		logger.print("Discriminator loaded", output_stream=sys.stdout)
	else:
		logger.print("No discriminator file found", output_stream=sys.stdout)
		






# scaling learning rate by number of GPUs.
generator_optimizer = Adam(2e-4 * hvd.size(),0.5)
discriminator_optimizer = Adam(2e-4 * hvd.size(),0.5)


# This is the training step function. It consums one batch and applies the gradient
def train_step(images):

	seed = tf.reshape(images[:,:, :, 0], (images.shape[0], GENERATE_SQUARE, GENERATE_SQUARE, 1))
	real = images[:,:, :, 1:3]
	
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = generator(seed, training=True)
		real_output = discriminator(real, training=True)
		fake_output = discriminator(generated_images, training=True)

		gen_loss = generator_loss(fake_output, real, generated_images)
		disc_loss = discriminator_loss(real_output, fake_output)

	gen_tape = hvd.DistributedGradientTape(gen_tape)
	disc_tape = hvd.DistributedGradientTape(disc_tape)


	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


		


	return gen_loss,disc_loss



# Main training loop
def train(dataset, epochs):
	start = time.time()
	batch =0	
	for epoch in range(epochs):
		epoch_start = time.time()

		gen_loss_list = []
		disc_loss_list = []

		for image_batch in dataset.take(DATASET_SIZE//(BATCH_SIZE*hvd.size())):
			losses = train_step(image_batch)
			gen_loss_list.append(losses[0])
			disc_loss_list.append(losses[1])
			if(batch==0 and epoch ==0):
				batch=1
				hvd.broadcast_variables(generator.variables, root_rank=0)
				hvd.broadcast_variables(discriminator.variables, root_rank=0)
				hvd.broadcast_variables(generator_optimizer.variables(), root_rank=0)
				hvd.broadcast_variables(discriminator_optimizer.variables(), root_rank=0)
		g_loss = sum(gen_loss_list) / len(gen_loss_list)
		d_loss = sum(disc_loss_list) / len(disc_loss_list)

		epoch_elapsed = time.time()-epoch_start
	
		if(hvd.rank()==0):
			save_images(OUTPUT_PATH, epoch,sample_images, generator, hvd.rank()==0)
			logger.print (f'Epoch: {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {epoch_elapsed}', output_stream=sys.stdout)
			logger.print(psutil.virtual_memory(), output_stream=sys.stdout)
			if(epoch%5==0):
				logger.print(f"Saving Model for Step {epoch}", output_stream=sys.stdout)
				generator.save(os.path.join(MODEL_PATH,f"color_generator_{epoch}.h5"))
				discriminator.save(os.path.join(MODEL_PATH,f"color_discriminator_{epoch}.h5"))

	elapsed = time.time()-start
	if(hvd.rank()==0):
		logger.print (f'Training time: {(elapsed)}', output_stream=sys.stdout)




logger.print("Starting Training", output_stream=sys.stdout)

# Starting the training
train(train_dataset, EPOCHS)

logger.print("Training Finished", output_stream=sys.stdout)


# saving the model to disk
if hvd.rank() == 0:
	logger.print("Saving Final Models", output_stream=sys.stdout)
	generator.save(GENERATOR_PATH_FINAL)
	discriminator.save(DISCRIMINATOR_PATH_FINAL)

