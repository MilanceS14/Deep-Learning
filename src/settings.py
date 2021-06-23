import os
import tensorflow as tf

root = '/home/ivanz/Desktop/chest_xray_data_and_models'

data_path = os.path.join(root, 'data_balanced')

train_path = os.path.join(data_path, 'train')
val_path = os.path.join(data_path, 'val')
test_path = os.path.join(data_path, 'test')

trained_models_path = os.path.join(root, 'trained_models', 'balanced_models')
if not os.path.exists(trained_models_path):
    os.mkdir(trained_models_path)

classes = ['NORMAL', 'PNEUMONIA']
batch_size = 32
image_size = 224
learning_rate = 1e-4

# Neki od mogucih algoritama za pretprocesiranje slika

preprocessing_algorithm = tf.keras.applications.resnet50.preprocess_input
# preprocessing_algorithm = tf.keras.applications.resnet_v2.preprocess_input
# preprocessing_algorithm = tf.keras.applications.efficientnet.preprocess_input
# preprocessing_algorithm = tf.keras.applications.vgg16.preprocess_input
# preprocessing_algorithm = tf.keras.applications.mobilenet_v2.preprocess_input