import tensorflow as tf
import numpy as np



data = tf.data.Dataset.from_tensor_slices([0, 1, 2, 3, 4, 5, 6, 7]).shuffle(8).batch(2).repeat()
for j in range(10):
	for i in data.take(2):
		print(i.numpy())