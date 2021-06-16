import numpy as np
from keras.engine.saving import load_model
from keras.preprocessing import image
import  os

from prettytable import PrettyTable

no_healthy=0
no_unhealthy=0
total=0
correct=0
t = PrettyTable(['Current', 'Predicted'])

model_path = 'model_cnn_1.19.h5'
# dimensions of images
#img_width, img_height = 128, 128

# load the trained model
model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for filename in os.listdir('dataset/single_prediction'):
   path='dataset/single_prediction/'+filename
   test_image = image.load_img(path, color_mode = "grayscale" ,target_size = (128, 128))
   test_image = image.img_to_array(test_image)
   test_image = np.expand_dims(test_image, axis = 0)
   total=total+1
   result = model.predict(test_image)
   if result[0][0] == 0:
    prediction = 'unhealthy'
    no_healthy += 1
   else:
    prediction = 'healthy'
    no_unhealthy += 1
   current_label = "healthy" if filename.startswith("healthy") else "unhealthy"
   t.add_row([current_label, prediction])
   if (prediction == current_label):
        correct = correct + 1
   #print('Result: ', result ,' pred:',prediction,' Filename: ', filename)



print(t)
print('Total: ', total, " Correct: ", correct, " Percentage: ", float(correct) / float(total))
print('Healthy number: ', no_healthy)
print('Unhealthy number: ', no_unhealthy)

# stop = timeit.default_timer()
# print("Stop time : ",stop)
# print('Time: ', stop - start)
# import matplotlib.pyplot as plt
#
# plt.plot(hist.history['accuracy'], label='train')
# plt.plot(hist.history['val_accuracy'], label='test')
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()
#
# plt.plot(hist.history['loss'], label='train')
# plt.plot(hist.history['val_loss'], label='test')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()
