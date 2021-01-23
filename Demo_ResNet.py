# transfer learning with ResNet network

import tensorflow as tf
from tensorflow.keras import datasets, layers
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
import numpy as np


# load data
(xtrain, ytrain), (xtest, ytest) = datasets.cifar10.load_data() 

classes   = np.unique(ytrain)
num_class = len(classes)

# split data into train & validate
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2, stratify=ytrain)

# normalize
xtrain = xtrain / 255.0
xtest  = xtest / 255.0
xvalid = xvalid / 255.0

# prepare network
input_shape    = (32, 32, 3) 
base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
# skip training phase in some layers
for layer in base_model.layers:
    layer.trainable = False
    
# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 1024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add a final softmax layer for classification
x = layers.Dense(num_class, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, x)


learning_rate = 0.001
model.compile(tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training & validation
history = model.fit(xtrain, ytrain, 
                    validation_data=(xvalid, yvalid), 
                    steps_per_epoch=100, 
                    epochs=10)

# testing 
loss, acc = model.evaluate(xtest, ytest)
pred      = model.predict(xtest)
pred      = np.argmax(pred, axis=1)
num_data  = len(ytest) 
correct   = 0
for i in range(num_data):
    if pred[i] == ytest[i]:
        correct += 1

acc_test = correct / num_data


