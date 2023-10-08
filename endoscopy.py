# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:25:20 2023

@author: Pushp Jain
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

def resize(input_image, input_mask):
   input_image = tf.image.resize(input_image, (128, 128), method="nearest")
   input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
   return input_image, input_mask

def augment(input_image, input_mask):
   if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       input_image = tf.image.flip_left_right(input_image)
       input_mask = tf.image.flip_left_right(input_mask)
   return input_image, input_mask

def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask

def load_image_train(image, mask):
   input_image = image
   input_mask = mask
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = augment(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)
   return input_image, input_mask

def load_image_test(image, mask):
   input_image = image
   input_mask = mask
   input_image, input_mask = resize(input_image, input_mask)
   input_image, input_mask = normalize(input_image, input_mask)
   return input_image, input_mask

# Define a function to load and preprocess the dataset
def load_and_preprocess_data(dataset_dir, input_shape):
    images = []
    masks = []

    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('_endo.png'):
                image_path = os.path.join(subdir, file)

                # Load and preprocess images and masks
                image = cv2.imread(image_path)
                image = cv2.resize(image, input_shape[:2])

                images.append(image)

            elif file.endswith('_endo_color_mask.png'):
                mask_path = os.path.join(subdir, file)

                mask = cv2.imread(mask_path)
                mask = cv2.resize(mask, input_shape[:2])

                masks.append(mask)


    return np.array(images), np.array(masks)


# Define data generators for training and validation
def data_generator(images, masks, batch_size):
    num_samples = len(images)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_images = images[batch_indices]
            batch_masks = masks[batch_indices]

            yield batch_images, batch_masks



# Load and preprocess the dataset
dataset_dir = 'C:/Users/DELL/Desktop/endoscopy/samples'
input_shape = (480, 854, 3)  # Adjust based on your dataset
num_classes = 3  # Number of classes in your dataset

images, masks = load_and_preprocess_data(dataset_dir, input_shape)




dataset = []

for image, mask in zip(images, masks):
    datapoint = {
        "image": image,
        "segmentation_mask": mask
    }
    dataset.append(datapoint)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Convert train_data into separate lists of images and masks
train_images = [datapoint["image"] for datapoint in train_data]
train_masks = [datapoint["segmentation_mask"] for datapoint in train_data]

# Convert the lists of images and masks into NumPy arrays
train_images = np.array(train_images)
train_masks = np.array(train_masks)

# Create TensorFlow Datasets for training data
train_dataset_images = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset_masks = tf.data.Dataset.from_tensor_slices(train_masks)

# Zip the datasets together to get a single dataset with pairs of (image, mask) for training
train_dataset = tf.data.Dataset.zip((train_dataset_images, train_dataset_masks))

# Apply the load_image_train function to the training dataset
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)

# Convert test_data into separate lists of images and masks
test_images = [datapoint["image"] for datapoint in test_data]
test_masks = [datapoint["segmentation_mask"] for datapoint in test_data]

# Convert the lists of images and masks into NumPy arrays for testing
test_images = np.array(test_images)
test_masks = np.array(test_masks)

# Create TensorFlow Datasets for testing data
test_dataset_images = tf.data.Dataset.from_tensor_slices(test_images)
test_dataset_masks = tf.data.Dataset.from_tensor_slices(test_masks)

# Zip the datasets together to get a single dataset with pairs of (image, mask) for testing
test_dataset = tf.data.Dataset.zip((test_dataset_images, test_dataset_masks))

# Apply the load_image_test function to the testing dataset
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)








def display(display_list):
 plt.figure(figsize=(15, 15))
 title = ["Input Image", "True Mask", "Predicted Mask"]
 for i in range(len(display_list)):
   plt.subplot(1, len(display_list), i+1)
   plt.title(title[i])
   plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
   plt.axis("off")
 plt.show()
sample_batch = next(iter(train_batches))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
display([sample_image, sample_mask])

"""
RGB to HEX: (Hexadecimel --> base 16)
This number divided by sixteen (integer division; ignoring any remainder) gives 
the first hexadecimal digit (between 0 and F, where the letters A to F represent 
the numbers 10 to 15). The remainder gives the second hexadecimal digit. 
0-9 --> 0-9
10-15 --> A-F

Example: RGB --> R=201, G=, B=

R = 201/16 = 12 with remainder of 9. So hex code for R is C9 (remember C=12)

Calculating RGB from HEX: #3C1098
3C = 3*16 + 12 = 60
10 = 1*16 + 0 = 16
98 = 9*16 + 8 = 152

"""
#Convert HEX to RGB array
# Try the following to understand how python handles hex values...
a=int('3C', 16)  #3C with base 16. Should return 60. 
print(a)
#Do the same for all RGB channels in each hex code to convert to RGB
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#') 
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation =  'FEDD3A'.lstrip('#') 
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip('#') 
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#') 
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

label = single_patch_mask

# Now replace RGB to integer values to be used as labels.
#Find pixels with combination of RGB for the above defined arrays...
#if matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)
 

print("Unique labels in label dataset are: ", np.unique(labels))


n_classes = len(np.unique(labels))
from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)



   
# unet_model = build_unet_model()
# unet_model.summary()
# dot_img_file = '/tmp/model_1.png'

# tf.keras.utils.plot_model(unet_model, to_file=dot_img_file, show_shapes=True)


# unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
#                   loss="sparse_categorical_crossentropy",
#                   metrics="accuracy")


# NUM_EPOCHS = 20
# TRAIN_LENGTH = 64
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
# VAL_SUBSPLITS = 5
# TEST_LENTH = 16
# VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS
# model_history = unet_model.fit(train_batches,
#                               epochs=NUM_EPOCHS,
#                               steps_per_epoch=STEPS_PER_EPOCH,
#                               validation_steps=VALIDATION_STEPS,
#                               validation_data=test_batches)

