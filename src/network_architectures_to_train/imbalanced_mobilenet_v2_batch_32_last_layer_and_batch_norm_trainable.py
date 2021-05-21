import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from src.train import train_neural_network
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, image_size, learning_rate


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'imbalanced_mobilenet_v2_batch_32_last_layer_and_batch_norm_trainable')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

mobilenet_backbone = MobileNetV2()
mobilenet_backbone.summary()

input_layer = mobilenet_backbone.layers[0].input
x = mobilenet_backbone.layers[-2].output
output = Dense(units=2, activation='softmax', name='predicted')(x)

model = Model(input_layer, output)
model.summary()

# Omoguciti samo poslednjem sloju i svim slojevima koji vrse batch normalizaciju da "uce"

for layer in model.layers[:-1]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False

model.summary()

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path, early_stopping_monitor='val_precision')