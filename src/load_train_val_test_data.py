from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .settings import classes, batch_size, train_path, val_path, test_path, preprocessing_algorithm, image_size

train_data = ImageDataGenerator(preprocessing_function=preprocessing_algorithm)\
                .flow_from_directory(directory=train_path, target_size=(image_size, image_size), classes=classes, batch_size=batch_size)
val_data = ImageDataGenerator(preprocessing_function=preprocessing_algorithm)\
                .flow_from_directory(directory=val_path, target_size=(image_size, image_size), classes=classes, batch_size=batch_size)
test_data = ImageDataGenerator(preprocessing_function=preprocessing_algorithm)\
                .flow_from_directory(directory=test_path, target_size=(image_size, image_size), classes=classes, batch_size=batch_size, shuffle=False)