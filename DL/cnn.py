from tensorflow.keras import layers, models


def jho(xtrain, ytrain, xvalid, yvalid, num_class, input_shape):
    # Parameters
    max_epochs  = 5
    batch_size  = 64 
    
    # build CNN
    model = models.Sequential([
        # 1st convolution
        layers.Conv2D(16, (3, 3), activation='relu', 
                      input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 2nd convolution
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 3rd convolution
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        # 1st fully connected
        layers.Dense(64, activation='relu'),
        
        # 2nd fully connected
        layers.Dense(num_class, activation='softmax')
    ])
       
    # complie model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # train & validate model
    history = model.fit(xtrain, ytrain, 
                        epochs=max_epochs, 
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(xvalid, yvalid))
    
    return model, history


