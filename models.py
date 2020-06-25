import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Dropout,
    Dense,
    Flatten,
    BatchNormalization,
    Activation,
    ZeroPadding2D,
    Concatenate,
    Add,
	LeakyReLU,
	UpSampling2D,
	Conv2D
)
from tensorflow import shape as shape_list
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
weight_init = tf.keras.initializers.TruncatedNormal(mean=0.1, stddev=0.02)

def build_discriminator(image_shape=(128,128, 2)):
	input_layer = Input(image_shape)
	x = Conv2D(32, 3, 2, "same")(input_layer)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.25)(x)
	x = attention(x, 32)
	x = Conv2D(64, 3, 2, "same")(x)
	x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.25)(x)
	x = Conv2D(64, 3, 2, "same")(x)
	x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.25)(x)
	x = Conv2D(64, 3, 2, "same")(x)
	x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.25)(x)
	x = attention(x, 64)
	x = Conv2D(128, 3, 2, "same")(x)
	x = BatchNormalization(momentum=0.8)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.25)(x)
	x = Flatten()(x)
	output_layer = Dense(1, activation="sigmoid")(x)

	return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="Discriminator")




def build_generator(image_shape=(128, 128, 1), filter_nums=32):
	input_layer = Input(shape=image_shape)
	x = down_resblock(input_layer, filter_nums)
	x_32 = x
	
	filter_nums*=2
	x = down_resblock(x, filter_nums, strides=2)
	x_64 = x
	
	filter_nums*=2
	x = down_resblock(x, filter_nums, strides=2)
	x_128 = x

	filter_nums*=2
	x = down_resblock(x, filter_nums, strides=2)
	x = down_resblock(x, filter_nums, strides=1)
	x = down_resblock(x, filter_nums, strides=1)

	filter_nums//=2
	x = up_resblock(x, filter_nums, add_layer=x_128, predicate = 1)

	filter_nums//=2
	x = up_resblock(x, filter_nums, add_layer=x_64, predicate = 1)

	filter_nums//=2
	x = up_resblock(x, filter_nums, add_layer=x_32, predicate = 1)

	x = attention(x, filter_nums)

	x = down_resblock(x, filter_nums, strides = 1)


	# Final CNN layer
	x = Conv2D(2, kernel_size=3, padding="same")(x)
	output_layer = Activation("tanh")(x)

	return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="Generator")




def down_resblock(x, filter_num, kernel_size=3, strides=1, padding="same", momentum=0.8, activation="relu"):
	x = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding=padding)(x)
	x = BatchNormalization(momentum=momentum)(x)
	x = Activation(activation)(x)
	return x

def up_resblock(x, filter_num, add_layer=None,kernel_size=3, padding="same", momentum=0.8, activation="relu", predicate=0 ):
	x = UpSampling2D(size=(2, 2))(x)
	x = Conv2D(filter_num, kernel_size=kernel_size, padding=padding)(x)
	x = BatchNormalization(momentum=momentum)(x)
	
	x = tf.cond(tf.less(tf.constant(predicate), tf.constant(1)), lambda: x, lambda: Add()([x, add_layer]))
	x = Activation("relu")(x)
	return x


def attention(x, ch):
	f = Conv2D(ch//8, 4, 2, use_bias=True, kernel_initializer=weight_init)(x)
	g = Conv2D(ch//8, 4, 2, use_bias=True, kernel_initializer=weight_init)(x)
	h = Conv2D(ch, 1, 1, use_bias=True, kernel_initializer=weight_init)(x)

	f = Flatten()(f)
	g = Flatten()(g)
	h = Flatten()(h)

	s = tf.matmul(g, f, transpose_b=True)

	beta = tf.nn.softmax(s)  

	o = tf.matmul(beta, h)

	gamma = tf.Variable(lambda: tf.keras.initializers.Zeros()(shape=[1]), trainable=True)

	o = tf.reshape(o, tf.shape(x))
	x = gamma * o + x

	return x	

def vgg_layers(layer_names, model):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = model
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model


def get_conv_layers(model):

	vgg = model
	layers = []
	for layer in vgg.layers:
		if("conv4" in layer.name or "conv3" in layer.name):
			layers.append(layer.name)
	return layers

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction= tf.keras.losses.Reduction.SUM)
mean_absolute = tf.keras.losses.MeanAbsoluteError(reduction= tf.keras.losses.Reduction.SUM)

def build_perceptual_model():
	vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
	name = get_conv_layers(vgg)
	model = vgg_layers(name, vgg)
	return model

def perceptual_loss(image_1, image_2, model):
	# assume image_1 and image_2 are in rgb color space with a range of [0, 1]
	preprocessed_image_1 = tf.keras.applications.vgg19.preprocess_input(image_1*255)
	preprocessed_image_2 = tf.keras.applications.vgg19.preprocess_input(image_2*255)

	output_image_1 = model(preprocessed_image_1)
	output_image_2 = model(preprocessed_image_2)
	output_images = list(zip(output_image_1, output_image_2))
	def temp_fn(x):
		temp = tf.math.l2_normalize(x[0], axis=3)-tf.math.l2_normalize(x[1], axis=3)
		temp = tf.math.sqrt(temp**2)
		temp = tf.reduce_sum(temp)
		return temp
	res = tf.vectorized_map(temp_fn, output_images) 	
	loss = tf.reduce_sum(res)/len(res)
	return loss

def perceptual_loss_gan_wrapper(images, generated_colors, model):
	# images: yuv color space with a range of [-0.5, 0.5]
	# images.shpae: (None, 128, 128, 3)
	# generated_colors: yuv color space a range of [-0.5, 0.5]
	# generated_colors.shape: (None, 128, 128, 2)
	last_dimension_axis = len(images.shape) - 1
	y, u, v = tf.split(images, 3, axis=last_dimension_axis)
	y = tf.add(y, 0.5)
	generated_images = tf.concat([y, generated_colors],axis = last_dimension_axis) 
	rgb_images = tf.image.yuv_to_rgb(tf.concat([y, u, v], axis=last_dimension_axis))
	rgb_generated_images = tf.image.yuv_to_rgb(generated_images)
	return perceptual_loss(rgb_images, rgb_generated_images, model)



def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, real_images, gen_images, lam, model):
    return cross_entropy(tf.ones_like(fake_output), fake_output) + lam * perceptual_loss_gan_wrapper(
        real_images, gen_images, model
    )

