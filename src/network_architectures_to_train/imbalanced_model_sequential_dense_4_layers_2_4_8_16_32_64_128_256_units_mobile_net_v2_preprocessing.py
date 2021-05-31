import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from src.train import train_neural_network
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, image_size, learning_rate

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'imbalanced_model_sequential_dense_4_layers_2_4_8_16_32_64_128_256_units_mobile_net_v2_preprocessing')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

model = Sequential([
    InputLayer((image_size, image_size, 3)),
    Flatten(),
    Dense(units=2, activation='relu'),
    Dense(units=4, activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=256, activation='relu'),
    Dense(units=2, activation='softmax')
])

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path)