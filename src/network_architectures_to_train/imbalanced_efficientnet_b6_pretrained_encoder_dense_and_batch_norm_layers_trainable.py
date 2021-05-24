from tensorflow.keras.applications.efficientnet import EfficientNetB6
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, Precision, Recall
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from src.load_train_val_test_data import train_data, val_data
from src.settings import trained_models_path, batch_size, learning_rate
from src.train import train_neural_network
import os

# putanja na koju treba snimiti model nakon obucavanja

model_save_path = os.path.join(trained_models_path, 'imbalanced_efficientnet_b6_pretrained_encoder_dense_and_batch_norm_layers_trainable')
if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

backbone = EfficientNetB6()
backbone.summary()

input_layer = backbone.layers[0].input
x = backbone.get_layer('top_dropout').output
output_layer = Dense(units=2, activation='softmax', name='predictions')(x)

model = Model(inputs=input_layer, outputs=output_layer)

for layer in model.layers[:-1]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = False

train_neural_network(model=model, train_data=train_data, val_data=val_data, batch_size=batch_size,
                    optimizer=Adam(learning_rate=learning_rate), loss=categorical_crossentropy, 
                    metrics=[categorical_accuracy, Precision(class_id=0), Recall(class_id=0)], epochs=1000, verbose=2, 
                    save_path=model_save_path, early_stopping_monitor='val_precision')