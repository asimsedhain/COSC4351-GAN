
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D, Concatenate, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.optimizers import Adam
from tensorflow.train import AdamOptimizer as Adam


GENERATE_RES = 4 # Generation resolution factor (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# generator code

def build_generator(channels=2, image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, 1)):

	input_layer = Input(shape=image_shape)
	x = Conv2D(64, kernel_size=3,activation="relu", padding="same")(input_layer)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)


	x = Conv2D(64,kernel_size=3, strides=1,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x_128_128 = Activation("relu")(x)

	x = Conv2D(128,kernel_size=3, strides=2,padding="same")(x_128_128)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)

	x = Conv2D(128,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x_64_64 = Activation("relu")(x)

	x = Conv2D(256,kernel_size=3, strides=2,padding="same")(x_64_64)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)

	x = Conv2D(256,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)

	x = Conv2D(256,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)



	# Output resolution, additional upsampling
	x = UpSampling2D(size=(2, 2))(x)
	x = Conv2D(128,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Concatenate()([x, x_64_64])
	x = Activation("relu")(x)

	x = Conv2D(128,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)


	x = UpSampling2D(size=(2,2))(x)
	x = Conv2D(64,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Concatenate()([x, x_128_128])
	x = Activation("relu")(x)

	x = Conv2D(64,kernel_size=3,padding="same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = Activation("relu")(x)

	# Final CNN layer
	x = Conv2D(2,kernel_size=3,padding="same")(x)
	output_layer = Activation("tanh")(x)

	return tf.keras.Model(inputs = input_layer, outputs=output_layer, name = "Generator")


# discrimator code

def build_discriminator(image_shape = (GENERATE_SQUARE, GENERATE_SQUARE, 2)):
	model = Sequential()

	model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
	model.add(ZeroPadding2D(padding=((0,1),(0,1))))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
	model.add(ZeroPadding2D(padding=((0,1),(0,1))))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	# model.add(Dropout(0.25))
	# model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
	# model.add(BatchNormalization(momentum=0.8))
	# model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	return model


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_absolute = tf.keras.losses.MeanAbsoluteError()

def discriminator_loss(real_output, fake_output):
	real_loss = cross_entropy(tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
	total_loss = real_loss + fake_loss
	return total_loss

def generator_loss(fake_output, real_images, gen_images ):
	return cross_entropy(tf.ones_like(fake_output), fake_output) + 100*mean_absolute(real_images, gen_images)
