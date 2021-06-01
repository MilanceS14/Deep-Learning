import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from src.train import train_neural_network
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, learning_rate

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'mobilenet_v2_batch_32_only_last_layer_trainable')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

mobilenet_backbone = MobileNetV2()

input_layer = mobilenet_backbone.layers[0].input
x = mobilenet_backbone.layers[-2].output
output = Dense(units=2, activation='softmax', name='predicted')(x)

model = Model(input_layer, output)

# Omoguciti samo poslednjem sloju da "uci"

for layer in model.layers[:-1]:
    layer.trainable = False

model.summary()

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path)