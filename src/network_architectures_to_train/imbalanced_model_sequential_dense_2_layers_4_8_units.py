import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from src.train import train_neural_network
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, image_size

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'imbalanced_model_sequential_dense_2_layers_4_8_units')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

model = Sequential([
    InputLayer((image_size, image_size, 3)),
    Flatten(),
    Dense(units=4, activation='relu'),
    Dense(units=4, activation='relu'),
    Dense(units=2, activation='softmax')
])

train_neural_network(model=model, train_data=train_data, val_data=val_data, 
                    optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=3, verbose=2, 
                    save_path=model_save_path, early_stopping_monitor='val_precision')