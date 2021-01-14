from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from DL.cnn import jho


# load data
(xtrain,ytrain), (xtest,ytest) = fashion_mnist.load_data()

classes   = np.unique(ytrain)
num_class = len(classes)

xtrain = xtrain.reshape(-1, 28,28, 1)
xtest  = xtest.reshape(-1, 28,28, 1)

# normalize
xtrain = xtrain / 255.0
xtest  = xtest / 255.0

# split data into train & validate
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2, stratify=ytrain)


# perform CNN
input_shape    = (28, 28, 1) 
model, history = jho(xtrain, ytrain, xvalid, yvalid, num_class, input_shape)

# test with unseen data
loss, acc = model.evaluate(xtest, ytest)
pred      = model.predict_classes(xtest)
# pred      = np.argmax(np.round(pred), axis=1)
num_data  = len(ytest) 
correct   = 0
for i in range(num_data):
    if pred[i] == ytest[i]:
        correct += 1

acc_test = correct / num_data


accuracy     = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss         = history.history['loss']
val_loss     = history.history['val_loss']
epochs       = range(len(accuracy))


# plot trianing & validation graph
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



