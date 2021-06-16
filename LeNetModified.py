# Image Classification

# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Activation, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from tensorflow.python.estimator import keras
from keras.layers import GlobalAveragePooling2D

# Initalize Modified Lenet
classifier = Sequential()

#1st set Conv-Relu-Pool
classifier.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(128, 128, 1), padding="same")) #initially (64,64,3)
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

classifier.add(Dropout(0.2, input_shape=(128, 128, 1)))

#2nd set Conv-Relu-Pool
classifier.add(Conv2D(filters=64, kernel_size=(5,5), padding="same"))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#3rd set Conv-Relu-Pool
classifier.add(Conv2D(filters=128, kernel_size=(5,5), padding="same"))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

#4th set Conv-Relu-Pool
#classifier.add(Conv2D(filters=128, kernel_size=(5,5), padding="same"))
#classifier.add(Activation("relu"))
#classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

#Flaten-Dense-Relu
classifier.add(Flatten())
classifier.add(Dense(units = 500, activation = 'relu'))

#dropout
classifier.add(Dropout(0.2, input_shape=(128, 128, 1))) #0.2

# softmax classifier

classifier.add(Dense(units=2, activation='softmax'))#1

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #binary_crossentropy
print(classifier.summary())

#saving in an csv file
csv_logger = CSVLogger("model_log_lenet_1.8.2.csv", append=True)

# Fit CNN to images


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical') #binary

history = classifier.fit_generator(
        train_set,
        steps_per_epoch=300, #9651/32=301
        epochs=100,
        validation_data=test_set,
        validation_steps=100,
        callbacks=[csv_logger])

#Plotting accuracy and loss for train and test

print(history.history.keys())

# summarize history for accuracy
plt.style.use("ggplot")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.style.use("ggplot")
plt.plot( history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#classifier.save('model1.h5')

from keras.models import model_from_json
model_json = classifier.to_json()
with open("model_lenet_1.8.2.json", "w") as json_file:
    json_file.write(model_json)

classifier.save_weights("model_lenet_1.8.2.h5")
print("Saved model to disk")


# saving HDF5 file
classifier.save('model_weights_lenet_1.8.2.h5')


