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



BATCH_SIZE = 4
BUFFER_SIZE = 1000
train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)




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


def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x


def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
       inputs = layers.Input(shape=(128,128,3))
       # encoder: contracting path - downsample
       # 1 - downsample
       f1, p1 = downsample_block(inputs, 64)
       # 2 - downsample
       f2, p2 = downsample_block(p1, 128)
       # 3 - downsample
       f3, p3 = downsample_block(p2, 256)
       # 4 - downsample
       f4, p4 = downsample_block(p3, 512)
       
       # 5 - bottleneck
       bottleneck = double_conv_block(p4, 1024)
       
       # decoder: expanding path - upsample
       # 6 - upsample
       u6 = upsample_block(bottleneck, f4, 512)
       # 7 - upsample
       u7 = upsample_block(u6, f3, 256)
       # 8 - upsample
       u8 = upsample_block(u7, f2, 128)
       # 9 - upsample
       u9 = upsample_block(u8, f1, 64)

       # outputs
       outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
       # unet model with Keras Functional API
       unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
       return unet_model
   
unet_model = build_unet_model()
unet_model.summary()
dot_img_file = '/tmp/model_1.png'

tf.keras.utils.plot_model(unet_model, to_file=dot_img_file, show_shapes=True)


unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")


NUM_EPOCHS = 20
TRAIN_LENGTH = 64
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VAL_SUBSPLITS = 5
TEST_LENTH = 16
VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS
model_history = unet_model.fit(train_batches,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches)

