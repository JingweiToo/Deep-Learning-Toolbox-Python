import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from DL import pre_data
from DL.alexnet import jho

# load data
(xtrain, ytrain), (xtest, ytest) = keras.datasets.cifar10.load_data()
# I only use small amount of images for training and validation
x1, x2, y1, y2 = train_test_split(xtrain, ytrain, test_size=0.2, stratify=ytrain)
# small amount of validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(x2, y2, test_size=0.1, stratify=y2)

del x1, x2, y1, y2

# prepare image data 
size       = 224
batch_size = 50 
train_ds, valid_ds, test_ds = pre_data.data(xtrain, xvalid, xtest, ytrain, yvalid, ytest, size, batch_size)

# train & validate model
num_class  = 10
max_epochs = 2 
model, history = jho(train_ds, valid_ds, num_class, max_epochs)

# test model with unseen data    
loss, acc = model.evaluate(test_ds)




