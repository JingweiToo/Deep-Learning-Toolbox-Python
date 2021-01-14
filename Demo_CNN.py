from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from DL.cnn import jho
import numpy as np


# load data
(xtrain, ytrain), (xtest, ytest) = datasets.cifar10.load_data() 

classes   = np.unique(ytrain)
num_class = len(classes)

# small amount of validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2, stratify=ytrain)

# normalize pixel values to be between 0 and 1
xtrain = xtrain / 255.0
xvalid = xvalid / 255.0
xtest  = xtest / 255.0

# visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plot data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(xtrain[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[ytrain[i][0]])
plt.show()

# perform CNN
input_shape    = (32, 32, 3)
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

# plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


