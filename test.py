import tensorflow as tf
import pathlib



tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

train = pathlib.Path("../val_set")

list_ds = tf.data.Dataset.list_files(str(train/'*'))

for i in list_ds.take(5):
	print(i.numpy())




