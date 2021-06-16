# Image Classification

# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, BatchNormalization, Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense

# Initalize CNN
from tensorflow.python.estimator import keras



classifier = Sequential()

# 1ST Add 2 convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(128, 128, 1), activation='relu', padding="same")) #initially (64,64,3)
#classifier.add(BatchNormalization())
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))
#classifier.add(BatchNormalization())

# Add pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 2ND Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 3RD Add 2 more convolution layers
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# 4TH Add 2 more convolution layers
#classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))#in model cnn_1.3 64
#classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"))

# Add max pooling layer
#classifier.add(MaxPooling2D(pool_size=(2,2)))

# Add global average pooling layer
classifier.add(GlobalAveragePooling2D())

#classifier.add(Dropout(0.2))


#add fully connetced layer
#classifier.add(Flatten())
#classifier.add(Dense(500))
#classifier.add(Activation("relu"))

#dropout
classifier.add(Dropout(0.5, input_shape=(128, 128, 1)))

# Add full connection
classifier.add(Dense(units=2, activation='softmax'))#1
#classifier.add(Dense(units=2, activation='softmax'))#add 2nd activation in model cnn_1.4
#classifier.add(Dense(units=2, activation='softmax'))#add 3rd activation in model cnn_1.5

#keras.models.Model().summary()
print(classifier.summary())

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #binary_crossentropy categorical_cross

#saving in an csv file
from keras.callbacks import CSVLogger

csv_logger = CSVLogger("model_log_cnn_1.19.csv", append=True)

# Fit CNN to images
from keras.preprocessing.image import ImageDataGenerator

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
        steps_per_epoch=300, #9651/32=301 238
        epochs=150,
        validation_data=test_set,
        validation_steps=150,
        callbacks=[csv_logger])

#Plotting accuracy and loss
import matplotlib.pyplot as plt
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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#classifier.save('model1.h5')

from keras.models import model_from_json
model_json = classifier.to_json()
with open("model_cnn_1.19.json", "w") as json_file:
    json_file.write(model_json)

classifier.save_weights("model_weights_cnn_1.19.h5")
print("Saved model to disk")


# Testing HDF5 file
classifier.save('model_cnn_1.19.h5')

#steps=100
#nr_correct, nr_guesses = test_accuracy(classifier, test_set, steps)
#print(nr_correct)
#print(nr_guesses)

# Test accuracy of classifier
#  def test_accuracy(classifier, test_set, steps):
#      num_correct = 0
#      num_guesses = 0
#      for i in range(steps):
#          a = test_set.next()
#          guesses = classifier.predict(a[0])
#          correct = a[1]
#          for index in range(len(guesses)):
#              num_guesses += 1
#              if round(guesses[index][0]) == correct[index]:
#                  num_correct += 1
#      return num_correct, num_guesses
