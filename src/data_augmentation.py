from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

normal_images_path = os.path.join(os.getcwd(), 'normal_images')
normal_images_augmented_path = os.path.join(os.getcwd(), 'normal_images_augmented')

datagen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.1)

i = 0
for batch in datagen.flow_from_directory(directory=normal_images_path, 
                                         target_size=(400, 400), 
                                         save_to_dir=normal_images_augmented_path, 
                                         save_prefix='aug', 
                                         save_format='png', 
                                         batch_size=32):

    i += 1
    if i > 133:
        break