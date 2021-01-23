# transfer learning with GoogLeNet network

import tensorflow as tf
from tensorflow.keras import datasets, layers
from keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
import numpy as np
from DL import pre_data


# load data
(xtrain, ytrain), (xtest, ytest) = datasets.cifar10.load_data() 

classes   = np.unique(ytrain)
num_class = len(classes)

# split data into train & validate
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2, stratify=ytrain)

# preparation
batch_size = 256 
train_ds, valid_ds, test_ds = pre_data.googlenet(xtrain, xvalid, xtest, ytrain, yvalid, ytest, batch_size)


# prepare network
input_shape = (75, 75, 3) 
base_model  = InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
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

# compile 
learning_rate = 0.001
model.compile(tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training & validation
max_epochs = 5 
history = model.fit(train_ds, 
                    validation_data=(valid_ds),
                    validation_freq=1,
                    epochs=max_epochs)

# testing 
loss, acc = model.evaluate(test_ds)


