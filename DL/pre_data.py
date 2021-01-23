import tensorflow as tf


def alexnet(xtrain, xvalid, xtest, ytrain, yvalid, ytest, batch_size):
    # parameters
    size_1 = 224
    size_2 = 224
    
    # better train & test format for tensorflow    
    train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    valid_ds = tf.data.Dataset.from_tensor_slices((xvalid, yvalid))
    test_ds  = tf.data.Dataset.from_tensor_slices((xtest, ytest))
    
    # size of training & testing data
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()
    test_ds_size  = tf.data.experimental.cardinality(test_ds).numpy()


    def process_image(image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (size_1, size_2))
        
        return image, label


    # Pre-processing & image resize
    train_ds = (train_ds
                .map(process_image)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True)
                )
    valid_ds = (valid_ds
                .map(process_image)
                .shuffle(buffer_size=valid_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True)
                )
    test_ds  = (test_ds
                .map(process_image)
                .shuffle(buffer_size=test_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True)
                )  
    
    return train_ds, valid_ds, test_ds



def googlenet(xtrain, xvalid, xtest, ytrain, yvalid, ytest, batch_size):
    # parameters
    size_1 = 75
    size_2 = 75
    
    # better train & test format for tensorflow    
    train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    valid_ds = tf.data.Dataset.from_tensor_slices((xvalid, yvalid))
    test_ds  = tf.data.Dataset.from_tensor_slices((xtest, ytest))
    
    # size of training & testing data
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()
    test_ds_size  = tf.data.experimental.cardinality(test_ds).numpy()


    def process_image(image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (size_1, size_2))
        
        return image, label


    # Pre-processing & image resize
    train_ds = (train_ds
                .map(process_image)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True)
                )
    valid_ds = (valid_ds
                .map(process_image)
                .shuffle(buffer_size=valid_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True)
                )
    test_ds  = (test_ds
                .map(process_image)
                .shuffle(buffer_size=test_ds_size)
                .batch(batch_size=batch_size, drop_remainder=True)
                )  
    
    return train_ds, valid_ds, test_ds


