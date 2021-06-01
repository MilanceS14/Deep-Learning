import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from src.train import train_neural_network
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, learning_rate

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, '3_conv_layers_filters_32_64_128_kernel_3_3_stride_1_mobile_net_v2_preprocessing_dropout')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    Dropout(0.25),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    Dropout(0.25),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    Dropout(0.25),
    Flatten(),
    Dense(units=2, activation='softmax')
])

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path)