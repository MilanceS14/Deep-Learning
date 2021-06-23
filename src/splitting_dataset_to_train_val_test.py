import random
import shutil
import os

dataset_balanced_normal_path = '/home/ivanz/Desktop/chest_xray_balanced/NORMAL'
dataset_balanced_pneumonia_path = '/home/ivanz/Desktop/chest_xray_balanced/PNEUMONIA'

if os.path.isdir(dataset_balanced_normal_path) and os.path.isdir(dataset_balanced_pneumonia_path):
    print('Putanje korektne!')

train_normal_path = '/home/ivanz/Desktop/chest_xray_data_and_models/data_balanced/train/NORMAL'
train_pneumonia_path = '/home/ivanz/Desktop/chest_xray_data_and_models/data_balanced/train/PNEUMONIA'

val_normal_path = '/home/ivanz/Desktop/chest_xray_data_and_models/data_balanced/val/NORMAL'
val_pneumonia_path = '/home/ivanz/Desktop/chest_xray_data_and_models/data_balanced/val/PNEUMONIA'

test_normal_path = '/home/ivanz/Desktop/chest_xray_data_and_models/data_balanced/test/NORMAL'
test_pneumonia_path = '/home/ivanz/Desktop/chest_xray_data_and_models/data_balanced/test/PNEUMONIA'

# postavljanje predodredjene podele

random.seed(12)

# EXTRACTING NORMAL IMAGES FULL PATHS 

normal_image_names_full_path = []

for normal_image_name in os.listdir(dataset_balanced_normal_path):
    normal_image_names_full_path.append(os.path.join(dataset_balanced_normal_path, normal_image_name))
    
len(normal_image_names_full_path)

# 70% slika iz NORMAL direktorijuma prebaciti u trening skup

normal_images_seventy_percents = random.sample(population=normal_image_names_full_path, k=2977)

if len(os.listdir(train_normal_path)) < 2977:
    for normal_image_path in normal_images_seventy_percents:
        shutil.copy(normal_image_path, train_normal_path)

        
normal_images_val_and_test_paths = []

for image_path in normal_image_names_full_path:
    if image_path not in normal_images_seventy_percents:
        normal_images_val_and_test_paths.append(image_path)

# 10% slika iz NORMAL direktorijuma prebaciti u validacioni skup

normal_images_ten_percents = random.sample(population=normal_images_val_and_test_paths, k=425)

if len(os.listdir(val_normal_path)) < 425:
    for normal_image_path in normal_images_ten_percents:
        shutil.copy(normal_image_path, val_normal_path)
        
rest_normal_images_path = []

for image_path in normal_images_val_and_test_paths:
    if image_path not in normal_images_ten_percents:
        rest_normal_images_path.append(image_path)

# preostalih 20% slika prebaciti iz NORMAL direktorijuma u test skup

if len(os.listdir(test_normal_path)) < 852:
    for image_path in rest_normal_images_path:
        shutil.copy(image_path, test_normal_path)

# EXTRACTING PNEUMONIA IMAGES FULL PATHS 

pneumonia_image_names_full_path = []

for pneumonia_image_name in os.listdir(dataset_balanced_pneumonia_path):
    pneumonia_image_names_full_path.append(os.path.join(dataset_balanced_pneumonia_path, pneumonia_image_name))
    
len(pneumonia_image_names_full_path)

# 70% slika iz PNEUMONIA direktorijuma prebaciti u trening skup

pneumonia_images_seventy_percents = random.sample(population=pneumonia_image_names_full_path, k=2991)

if len(os.listdir(train_pneumonia_path)) < 2991:
    for pneumonia_image_path in pneumonia_images_seventy_percents:
        shutil.copy(pneumonia_image_path, train_pneumonia_path)

        
pneumonia_images_val_and_test_paths = []

for image_path in pneumonia_image_names_full_path:
    if image_path not in pneumonia_images_seventy_percents:
        pneumonia_images_val_and_test_paths.append(image_path)

# 10% slika iz PNEUMONIA direktorijuma prebaciti u validacioni skup

pneumonia_images_ten_percents = random.sample(population=pneumonia_images_val_and_test_paths, k=427)

if len(os.listdir(val_pneumonia_path)) < 427:
    for pneumonia_image_path in pneumonia_images_ten_percents:
        shutil.copy(pneumonia_image_path, val_pneumonia_path)
        
rest_pneumonia_images_path = []

for image_path in pneumonia_images_val_and_test_paths:
    if image_path not in pneumonia_images_ten_percents:
        rest_pneumonia_images_path.append(image_path)

# preostalih 20% slika prebaciti iz PNEUMONIA direktorijuma u test skup

if len(os.listdir(test_pneumonia_path)) < 855:
    for image_path in rest_pneumonia_images_path:
        shutil.copy(image_path, test_pneumonia_path)