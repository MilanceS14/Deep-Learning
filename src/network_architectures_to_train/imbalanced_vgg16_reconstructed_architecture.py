from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, learning_rate
from src.train import train_neural_network
import os

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'imbalanced_vgg16_reconstructed_architecture')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

model = Sequential([
    InputLayer((224, 224, 3), name='input_1'),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1'),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv2'),
    MaxPooling2D(pool_size=(2, 2), name='block1_pool'),
    Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv1'),
    Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv2'),
    MaxPooling2D(pool_size=(2, 2), name='block2_pool'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv1'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv2'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='block3_conv3'),
    MaxPooling2D(pool_size=(2, 2), name='block3_pool'),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv1'),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv2'),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block4_conv3'),
    MaxPooling2D(pool_size=(2, 2), name='block4_pool'),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv1'),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv2'),
    Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='block5_conv3'),
    MaxPooling2D(pool_size=(2, 2), name='block5_pool'),
    Flatten(name='flatten'),
    Dense(units=4096, activation='relu', name='fc1'),
    Dense(units=4096, activation='relu', name='fc2'),
    Dense(units=2, activation='softmax', name='predictions'),
])

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path)