from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, learning_rate
from src.train import train_neural_network
import os

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'resnet50_pretrained_encoder_dense_layer_trainable')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

backbone = ResNet50()

input_layer = backbone.get_layer('input_1').input
x = backbone.get_layer('avg_pool').output
output_layer = Dense(units=2, activation='softmax', name='predictions')(x)

model = Model(inputs=input_layer, outputs=output_layer)

for layer in model.layers[:-1]:
    layer.trainable = False

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path)