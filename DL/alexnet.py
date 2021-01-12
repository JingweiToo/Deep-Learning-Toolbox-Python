#[2017]-"ImageNet Classification with Deep Convolutional Neural Networks" 

import tensorflow as tf
from tensorflow.keras import layers, models


# desired input_shape=(224, 224, 3)

def jho(train_ds, valid_ds, num_class, max_epochs):
    
    # AlexNet model
    model = models.Sequential([
        # 1st convolution 
        layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        
        # 2nd convolution
        layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        
        # 3rd convolution
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        
        # 4th convolution
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        
        # 5th convolution
        layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        
        # flatten 
        layers.Flatten(),
        
        # 1st fully connected 
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        
        # 2nd fully connected 
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        
        # 3rd fully connected softmax 
        layers.Dense(num_class, activation='softmax')
    ])
    
    # build model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])
    
    # train & validate model
    history = model.fit(train_ds,
                        epochs=max_epochs,
                        validation_data=valid_ds,
                        validation_freq=1)
    
    return model, history 



