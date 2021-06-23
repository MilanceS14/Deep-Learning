import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, SeparableConv2D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from src.train import train_neural_network
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, image_size, learning_rate

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'test_architecture_from_kaggle_1_mobilenet_v2_preprocessing')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

model = Sequential()

# 1st conv
model.add(Conv2D(filters = 16, kernel_size = (3, 3), input_shape = (image_size, image_size, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size = (2, 2)))

# 2nd conv
model.add(SeparableConv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))

# 3rd conv
model.add(SeparableConv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

# 4th conv
model.add(SeparableConv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

# 5th conv
model.add(SeparableConv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(SeparableConv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

# fully connected layer
model.add(Flatten())
model.add(Dense(units = 512, activation = 'relu'))
model.add(Dropout(rate = 0.7))
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(rate = 0.3))

# output layer
model.add(Dense(units = 2, activation = 'softmax'))

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path)