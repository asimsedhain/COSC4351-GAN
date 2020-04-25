from utils import get_samples
from utils import save_images
import tensorflow as tf
import numpy as np
import os
import sys

# Training data directory
TRAINING_DATA_PATH = "../val_set"

# All the output and models will be saved inside the checkpoint path
CHECKPOINT_PATH = "./test_horovod"

# Sample images will be stored in the output path
OUTPUT_PATH = os.path.join(CHECKPOINT_PATH, "output") 

# Path for the model. It is inside the checkpoint directory
MODEL_PATH = os.path.join(CHECKPOINT_PATH,"Models")


# Path for the final models to be saved to after training
GENERATOR_PATH_FINAL = os.path.join(MODEL_PATH,"color_generator_final.h5")


tf.print("Loading model from memory", output_stream=sys.stdout)
if os.path.isfile(GENERATOR_PATH_FINAL):
	generator = tf.keras.models.load_model(GENERATOR_PATH_FINAL)
	tf.print("Generator loaded", output_stream=sys.stdout)
else:
	tf.print("Generator could not be loaded", output_stream=sys.stdout)
	raise FileExistsError()

sample_images = get_samples(TRAINING_DATA_PATH)

save_images(OUTPUT_PATH, "sample",sample_images, generator, True)
