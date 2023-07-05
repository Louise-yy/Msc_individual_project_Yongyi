import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import to_categorical
from keras import backend as K

K.clear_session()
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set data path and file name
image_dir = "/media/louise/UBUNTU 20_0/yy/DL/data"
label_file = "file/label.csv"

# Read the label file, get the image file name and corresponding label
df = pd.read_csv(label_file)
df_image_filenames = df["stop_frame"].tolist()
# df_labels = df["all_noun_classes"].tolist()
df_labels = df["all_noun_classes"].astype(str)

mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(df_labels)


def preprocess_image(image):
    resized_image = cv2.resize(image, (100, 100))

    normalized_image = resized_image.astype(np.float32)

    normalized_image /= 255.0

    return normalized_image


# Load and process image data
image_data = []
for filename in df_image_filenames:
    # The full path to the image file
    image_path = os.path.join(image_dir, filename)

    image = cv2.imread(image_path)

    # Perform image preprocessing
    image = preprocess_image(image)

    # Add image data to the data list
    image_data.append(image)

# Split image data and labels into training and testing sets
# train_data, test_data = train_test_split(image_data, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=0.2, random_state=42)

# def to_categorical_wrapper(label):
#     return tf.py_function(to_categorical, [label, 20], tf.int32)


# # A data generator that creates a training set
# train_image_data = tf.data.Dataset.from_generator(
#     lambda: (image for image, _ in train_data),
#     output_signature=tf.TensorSpec(shape=(100, 100, 3), dtype=tf.float32)
# )
# train_label_data = tf.data.Dataset.from_generator(
#     # Use lambda function to generate label data, where each element is a label
#     lambda: (label for _, label in train_data),
#     # Specifies the shape and data type of the tensor output by the generator
#     # the shape of the label is () and the data type is int32
#     output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
# )
# # The to_categorical_wrapper function converts label data into a one-hot encoding form train_label_data =
# # train_label_data.map(to_categorical_wrapper)
# #
# # The image data and label data are paired, and the paired data is
# # divided into batches, each batch contains 32 images and corresponding labels
# train_dataset = tf.data.Dataset.zip((train_image_data, train_label_data)).batch(4)

# # A data generator that creates a testing set
# test_image_data = tf.data.Dataset.from_generator(
#     lambda: (image for image, _ in test_data),
#     output_signature=tf.TensorSpec(shape=(150, 150, 3), dtype=tf.float32)
# )
# test_label_data = tf.data.Dataset.from_generator(
#     lambda: (label_encoder.transform([label])[0] for _, label in test_data),
#     output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
# )
# test_label_data = test_label_data.map(to_categorical_wrapper)
# test_dataset = tf.data.Dataset.zip((test_image_data, test_label_data)).conv_base = VGG16(weights='imagenet',
#                   include_top=False,
#                   input_shape=(150, 150, 3))

# 创建 TensorFlow 的数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# 对训练集和测试集进行批处理和混洗
batch_size = 8
train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Built model
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(100, 100, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))
model.summary()

conv_base.trainable = False

# Compile model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)