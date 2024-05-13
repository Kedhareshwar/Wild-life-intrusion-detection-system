# python librairies installation
# display, transform, read, split ...
import numpy as np
import cv2 as cv
import os
import splitfolders
import matplotlib.pyplot as plt
# tensorflow
import tensorflow.keras as keras
import tensorflow as tf
# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# model / neural network
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
"""
Step 2 - Data preprocessing
To use your data (images), you have to pre-process them.
Create Keras data generators
"""
import splitfolders
# split data in a new folder named data-split
splitfolders.ratio("C:/Users/sreer/Downloads/Page1", 
output="C:/Users/sreer/Downloads/Animal Split/", seed=1337, ratio=(0.7, 0.2, 0.1), 
group_prefix=None, move=False)
datagen = ImageDataGenerator()
# define classes name
class_names = ['Calm','Bear','Fox','Hyena','Lion','Tiger','Wolf']
# training data
train_generator = datagen.flow_from_directory(
 directory="C:/Users/sreer/Downloads/Animal Split/train",
 classes = class_names,
 target_size=(224, 224),
 batch_size=32,
 class_mode="sparse",)
# validation data
valid_generator = datagen.flow_from_directory(
 directory="C:/Users/sreer/Downloads/Animal Split/validation",
 classes = class_names,
 target_size=(224, 224),
 batch_size=32,
 class_mode="sparse",)
# test data
test_generator = datagen.flow_from_directory(
 directory="C:/Users/sreer/Downloads/Animal Split/test",
 classes = class_names,
 target_size=(224, 224),
 batch_size=32,
 class_mode="sparse",)
""" 
Step 3 - Build the model
The first step is to build the model, using *ResNet50*.
"""
# ResNet50 model
resnet_50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
for layer in resnet_50.layers:
 layer.trainable = False
# build the entire model
x = resnet_50.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(7, activation='softmax')(x)
model = Model(inputs = resnet_50.input, outputs = predictions)
"""
Step 4 - Train the model
*Adam* optimizer is used to train the model over *10 epochs*. It is enough by using 
Transfer Learning.
The loss is calculated with the *sparse_categorical_crossentropy* function.
"""
# define training function
def trainModel(model, epochs, optimizer):
 batch_size = 32
 model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
metrics=["accuracy"])
 return model.fit(train_generator, validation_data=valid_generator, epochs=epochs, 
batch_size=batch_size)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
# launch the training
model_history = trainModel(model = model, epochs = 20, optimizer = optimizer)
# Display *loss* curves:
loss_train_curve = model_history.history["loss"]
loss_val_curve = model_history.history["val_loss"]
plt.plot(loss_train_curve, label = "Train")
plt.plot(loss_val_curve, label = "Validation")
plt.legend(loc = 'upper right')
plt.title("Loss")
plt.show()
# Display *accuracy* curves:
acc_train_curve = model_history.history["accuracy"]
acc_val_curve = model_history.history["val_accuracy"]
plt.plot(acc_train_curve, label = "Train")
plt.plot(acc_val_curve, label = "Validation")
plt.legend(loc = 'lower right')
plt.title("Accuracy")
plt.show()
# Save the model
model.save('D:/Projects/Major/models from spyder/forestmodel.h5')
"""
Step 5 - Evaluate the model
The model is evaluated on test data.
"""
from tensorflow.keras.models import load_model
# Load your saved model
modelnew = load_model('D:/Projects/Major/models from spyder/forestmodel.h5')
test_loss, test_acc = model.evaluate(test_generator)
print("The test loss is: ", test_loss)
print("The best accuracy is: ", test_acc*100)
"""
Step 6 - Deploy the model
The model is deployed on live camera feed.
"""
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
# Define the classes
classes = ['Calm', 'Bear', 'Fox', 'Hyena', 'Lion', 'Tiger', 'Wolf']
# Function to preprocess the image
def preprocess_img(img):
 img = cv2.resize(img, (224, 224))
 img = image.img_to_array(img)
 img = np.expand_dims(img, axis=0)
 img = preprocess_input(img)
 return img
# Accessing live camera feed
cap = cv2.VideoCapture(0)
while True:
 ret, frame = cap.read()
 if not ret:
 break
 # Preprocess the frame
 processed_frame = preprocess_img(frame)
 # Predict the class
 prediction = modelnew.predict(processed_frame)
 predicted_class = classes[np.argmax(prediction)]
 # Display the result
 cv2.putText(frame, predicted_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 
(255, 0, 0), 4)
 cv2.imshow('Animal Classification', frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
cap.release()
cv2.destroyAllWindows()