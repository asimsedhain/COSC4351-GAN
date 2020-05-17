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
weight_regularizer = None

def build_discriminator(image_shape=(128,128, 2)):
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same")
    )
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
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
    model.add(Dense(1, activation="sigmoid"))

    return model


# def build_generator(channels=2, image_shape=(128, 128, 1)):

#     input_layer = Input(shape=image_shape)
#     x = Conv2D(64, kernel_size=3, activation="relu", padding="same")(input_layer)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)
#     x = Conv2D(64, kernel_size=3, strides=1, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x_128_128 = Activation("relu")(x)

#     x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x_128_128)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)

#     x = Conv2D(128, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x_64_64 = Activation("relu")(x)

#     x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x_64_64)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)

#     x = Conv2D(256, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)

#     x = Conv2D(256, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)
#     # Output resolution, additional upsampling
#     x = UpSampling2D(size=(2, 2))(x)
#     x = Conv2D(128, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Concatenate()([x, x_64_64])
#     x = Activation("relu")(x)

#     x = Conv2D(128, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)

#     x = UpSampling2D(size=(2, 2))(x)
#     x = Conv2D(64, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Concatenate()([x, x_128_128])
#     x = Activation("relu")(x)

#     x = Conv2D(64, kernel_size=3, padding="same")(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation("relu")(x)

#     # Final CNN layer
#     x = Conv2D(2, kernel_size=3, padding="same")(x)
#     output_layer = Activation("tanh")(x)

#     return tf.keras.Model(inputs=input_layer, outputs=output_layer, name="Generator")


def build_generator(channels=2, image_shape=(128, 128, 1), filter_nums=32):
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


def attention(x, filters, is_training, scope='attention', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        f = spectral_conv2d(x, filters // 8, kernel_size=1, stride=1, padding='VALID', is_training=is_training,
                            scope='f_conv')  # [bs, h, w, c']
        g = spectral_conv2d(x, filters // 8, kernel_size=1, stride=1, padding='VALID', is_training=is_training,
                            scope='g_conv')  # [bs, h, w, c']
        h = spectral_conv2d(x, filters, kernel_size=1, stride=1, padding='VALID', is_training=is_training,
                            scope='h_conv')  # [bs, h, w, c]

        f_flatten = flatten(f)  # [bs, h*w=N, c]
        g_flatten = flatten(g)

        s = tf.matmul(g_flatten, f_flatten, transpose_b=True)  # [bs, N, N]
        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, flatten(h))  # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=shape_list(x))  # [bs, h, w, c]
        x = gamma * o + x

    return x





def spectral_conv2d(x, filters, kernel_size, stride, is_training, padding='SAME', use_bias=True, scope='conv2d'):
    with tf.variable_scope(scope):
        w = tf.get_variable("conv_w",
                            shape=[kernel_size, kernel_size, shape_list(x)[-1], filters],
                            initializer=weight_init,
                            regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=x,
                         filter=spectral_norm(w, is_training),
                         strides=[1, stride, stride, 1],
                         padding=padding)
        if use_bias:
            bias = tf.get_variable("conv_bias", [filters], initializer=tf.constant_initializer(0.0))
            x = tf.nn.bias_add(x, bias)
    return x


def spectral_norm(w, is_training, iteration=1):
    w_shape = shape_list(w)
    w = tf.reshape(w, [-1, w_shape[-1]])  # [N, output_filters] kernel_size*kernel_size*input_filters = N

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(),
                        trainable=False)  # [1, output_filters]

    u_norm = u
    v_norm = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_norm, w, transpose_b=True)  # [1, N]
        v_norm = l2_norm(v_)

        u_ = tf.matmul(v_norm, w)  # [1, output_filters]
        u_norm = l2_norm(u_)

    # Au=λ1u  u⊤Au=λ1u⊤u=λ1
    sigma = tf.matmul(tf.matmul(v_norm, w), u_norm, transpose_b=True)  # [1,1]
    w_norm = w / sigma

    # Update estimated 1st singular vector while training
    with tf.control_dependencies([tf.cond(is_training,
                                          true_fn=lambda: u.assign(u_norm), false_fn=lambda: u.assign(u))]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(x):
    tensor_shape = shape_list(x)
    return tf.reshape(x, shape=[tensor_shape[0], -1, tensor_shape[-1]])




	


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction= tf.keras.losses.Reduction.NONE)
mean_absolute = tf.keras.losses.MeanAbsoluteError(reduction= tf.keras.losses.Reduction.NONE)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output, real_images, gen_images, lam):
    return cross_entropy(tf.ones_like(fake_output), fake_output) + lam * mean_absolute(
        real_images, gen_images
    )

