import tensorflow as tf


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

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, (128, 128))
    img = img[tf.newaxis, :]
    return img



def perceptual_loss(image_1, image_2, model):
	# assume image_1 and image_2 are in rgb color space with a range of [0, 1]
	preprocessed_image_1 = tf.keras.applications.vgg19.preprocess_input(image_1*255)
	preprocessed_image_2 = tf.keras.applications.vgg19.preprocess_input(image_2*255)

	output_image_1 = model(preprocessed_image_1)
	output_image_2 = model(preprocessed_image_2)

	res = []
	for output_layer_1, output_layer_2 in zip(output_image_1, output_image_2):
		temp = tf.math.l2_normalize(output_layer_1, axis=3)-tf.math.l2_normalize(output_layer_2, axis=3)
		temp = tf.math.sqrt(temp**2)
		temp = tf.reduce_sum(temp)
		res.append(temp)
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

mean_loss = tf.keras.losses.MeanAbsoluteError(reduction= tf.keras.losses.Reduction.SUM)

# vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
# name = get_conv_layers(vgg)
# model = vgg_layers(name, vgg)

# reference = load_img("./example_01.jpg")
# liquify = load_img("./example_02.jpg")
# blur = load_img("./example_03.jpg")
# more_blur = load_img("./example_04.jpg")
# more_more_blur = load_img("./example_05.jpg")

# print(f"Perceptual Distance to Liquify: {perceptual_loss(reference, liquify, model)}")
# print(f"Perceptual Distance to blur: {perceptual_loss(reference, blur, model)}")
# print(f"Perceptual Distance to more blur: {perceptual_loss(reference, more_blur, model)}")
# print(f"Perceptual Distance to more more blur: {perceptual_loss(reference, more_more_blur, model)}")

# print(f"Mean Distance to Liquify: {mean_loss(reference, liquify)}")
# print(f"Mean Distance to blur: {mean_loss(reference, blur)}")
# print(f"Mean Distance to more blur: {mean_loss(reference, more_blur)}")
# print(f"Mean Distance to more more blur: {mean_loss(reference, more_more_blur)}")




# original = tf.zeros((16, 128, 128, 3))
# fake = tf.zeros((16, 128, 128, 2))

# loss = perceptual_loss_gan_wrapper(original, fake, model)
# print(loss)



# y, u, v = tf.split(original, 3, axis=3)
# temp = tf.concat([y, tf.zeros((16, 128, 128, 2))], axis=3)


# print(temp.shape)


from models import generator_loss

loss = generator_loss(tf.zeros((16, 128, 128, 3)), tf.ones((16, 128, 128, 3)), tf.ones((16, 128, 128, 2)), 100)

reduced_loss = tf.reduce_sum(loss) * (1.0 / 128)
print(reduced_loss)


