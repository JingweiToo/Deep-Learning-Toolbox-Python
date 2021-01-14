from tensorflow import keras
from sklearn.model_selection import train_test_split
from DL import pre_data
from DL.alexnet import jho
import numpy as np


# load data
(xtrain, ytrain), (xtest, ytest) = keras.datasets.cifar10.load_data()

classes   = np.unique(ytrain)
num_class = len(classes)


# I only use small amount of images for training and validation
x1, x2, y1, y2 = train_test_split(xtrain, ytrain, test_size=0.2, stratify=ytrain)
# small amount of validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(x2, y2, test_size=0.1, stratify=y2)

del x1, x2, y1, y2

# prepare image data 
batch_size = 64 
train_ds, valid_ds, test_ds = pre_data.alexnet(xtrain, xvalid, xtest, ytrain, yvalid, ytest, batch_size)

# train & validate model
model, history = jho(train_ds, valid_ds, num_class)

# test model with unseen data    
loss, acc = model.evaluate(test_ds)






