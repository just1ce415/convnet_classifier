import keras
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

class ResNet50:
    def __init__(self):
        self.model = None

    def get_residual_model(self):
        return self.model

    def print_summary(self):
        print(self.model.summary())

    
    def __identity_block(self, X, f, filters, training=True, initializer=random_uniform):
        """
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        training -- True: Behave in training mode
                    False: Behave in inference mode
        initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
        
        Returns:
        X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
        """
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value to do a shortcut.
        X_shortcut = X
        
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X, training = training) # Default axis
        X = Activation('relu')(X)
        
        ## Set the padding = 'same'
        X = Conv2D(filters=F2, kernel_size=f, strides=(1,1), padding='same', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)

        ## Set the padding = 'valid'
        X = Conv2D(filters=F3, kernel_size=1, strides=(1,1), padding='valid', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        
        # Add shortcut
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X


    def __convolutional_block(self, X, f, filters, s = 2, training=True, initializer=glorot_uniform):
        """
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        s -- Integer, specifying the stride to be used
        training -- True: Behave in training mode
                    False: Behave in inference mode
        initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                    also called Xavier uniform initializer.
        
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
        
        # Retrieve Filters
        F1, F2, F3 = filters
        
        # Save the input value
        X_shortcut = X

        # First component of main path glorot_uniform(seed=0)
        X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
        X = BatchNormalization(axis = 3)(X, training=training)
        X = Activation('relu')(X)

        X = Conv2D(filters=F2, kernel_size=f, strides=(1,1), padding='same', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = Activation('relu')(X)

        X = Conv2D(filters=F3, kernel_size=1, strides=(1,1), padding='valid', kernel_initializer=initializer(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        
        ##### SHORTCUT PATH #####
        X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s,s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)
        

        # Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        
        return X

    def __data_augmenter(self):
        data_augmentation = keras.Sequential()
        data_augmentation.add(RandomFlip(mode='horizontal'))
        data_augmentation.add(RandomRotation(0.2))
        return data_augmentation

    def build_model(self, input_shape = (256, 265, 3), classes = 5):
        """
        Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """
        
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Augment data
        data_augmentation = self.__data_augmenter()
        X = data_augmentation(X_input)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X)
        
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.__convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
        X = self.__identity_block(X, 3, [64, 64, 256])
        X = self.__identity_block(X, 3, [64, 64, 256])

        ## Stage 3 (≈4 lines)
        ## The convolutional block uses three sets of filters of size [128,128,512], "f" is 3 and "s" is 2.
        ## The 3 identity blocks use three sets of filters of size [128,128,512] and "f" is 3.
        X = self.__convolutional_block(X, f=3, filters=[128,128,512], s=2)
        X = self.__identity_block(X, f=3, filters=[128,128,512])
        X = self.__identity_block(X, f=3, filters=[128,128,512])
        X = self.__identity_block(X, f=3, filters=[128,128,512]) 
        
        ## Stage 4 (≈6 lines)
        ## The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3 and "s" is 2.
        ## The 5 identity blocks use three sets of filters of size [256, 256, 1024] and "f" is 3.
        X = self.__convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
        X = self.__identity_block(X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X, f=3, filters=[256, 256, 1024]) 
        X = self.__identity_block(X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X, f=3, filters=[256, 256, 1024])
        X = self.__identity_block(X, f=3, filters=[256, 256, 1024])

        ## Stage 5 (≈3 lines)
        ## The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3 and "s" is 2.
        ## The 2 identity blocks use three sets of filters of size [512, 512, 2048] and "f" is 3.
        X = self.__convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
        X = self.__identity_block(X, f=3, filters=[512, 512, 2048])
        X = self.__identity_block(X, f=3, filters=[512, 512, 2048])

        ## AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D(pool_size=(2,2))(X)

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
        
        # Create model
        model = Model(inputs = X_input, outputs = X)

        self.model = model

        # compile model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class InceptionNet:
    pass