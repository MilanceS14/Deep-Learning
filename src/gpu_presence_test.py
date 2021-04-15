# provera da li tensorflow prepoznaje graficku karticu

import tensorflow as tf

l = tf.config.list_physical_devices("GPU")
print(l)