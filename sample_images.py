from utils import get_sample
from utils import save_images
import tensorflow as tf
import numpy as np
import os
import sys




# Training data directory
TRAINING_DATA_PATH = "../val_set"

# All the output and models will be saved inside the checkpoint path
CHECKPOINT_PATH = "./output/singularity_multigpu"

# Sample images will be stored in the output path
OUTPUT_PATH = os.path.join(CHECKPOINT_PATH, "sample") 

# Path for the model. It is inside the checkpoint directory
MODEL_PATH = os.path.join(CHECKPOINT_PATH,"models")


# Path for the final models to be saved to after training
GENERATOR_PATH_FINAL = os.path.join(MODEL_PATH,"color_generator_95.h5")


tf.print("Loading model from memory", output_stream=sys.stdout)
if os.path.isfile(GENERATOR_PATH_FINAL):
	generator = tf.keras.models.load_model(GENERATOR_PATH_FINAL)
	tf.print("Generator loaded", output_stream=sys.stdout)
else:
	tf.print("Generator could not be loaded", output_stream=sys.stdout)
	raise FileExistsError()

tf.print("Sampling Started...", output_stream=sys.stdout)
for j in range(30):
	sample_images = get_sample(TRAINING_DATA_PATH)
	sample_images = tf.convert_to_tensor([i.numpy() for i in sample_images.take(1)])

	save_images(OUTPUT_PATH, f"sample_{j}",sample_images, generator)
