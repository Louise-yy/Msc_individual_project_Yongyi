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
label_file = "file/label_special(2).csv"

# Read the label file, get the image file name and corresponding label
df = pd.read_csv(label_file)
df_image_filenames = df["stop_frame"].tolist()
# df_labels = df["all_noun_classes"].tolist()
df_labels = df["all_noun_classes"].astype(str)

mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(df_labels)

num_classes = 20
binary_encoded = np.zeros((len(df_labels), num_classes), dtype=int)
for i, label_list in enumerate(df_labels):
    for label in label_list:
        binary_encoded[i, label - 1] = 1


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
model.add(layers.Dense(20, activation='sigmoid'))
model.summary()

conv_base.trainable = False

# Compile model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)
