import keras
from sklearn.manifold import trustworthiness
import tensorflow as tf
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

from models import ResNet50, FFVGG

class ModelManager:
    def __init__(self, shape=256, labels=['1', '2', '3', '4', '5']):
        """
        Arguments:
        shape - shape of the input image represented as single integer, so height and width are the same.
        classes - number of classes

        Fields:
        model - keras.Model instance
        train_dataset - keras.Dataset instance for training
        validation_dataset - keras.Dataset instance for validation.
        """
        self.model = None
        self.train_dataset = None
        self.validation_dataset = None
        self.shape = shape
        self.classes = len(labels)
        self.classes_dict = {}
        for i, label in enumerate(labels):
            self.classes_dict[i] = label
        self.__loaded = False

    def create_residual_model(self):
        """
        Creates ResNet50 model (keras.Model instance)
        """
        if self.__loaded:
            print("Pretrained model is already loaded.")
            return
        resnet = ResNet50()
        resnet.build_model((self.shape, self.shape, 3), self.classes)
        self.model = resnet.get_residual_model()

    def create_ffvgg_model(self):
        """
        Creates Fast Fourier VGG model
        """
        if self.__loaded:
            print("Pretrained model is already loaded.")
            return
        ffvgg = FFVGG()
        ffvgg.build_model((self.shape, self.shape, 3), self.classes)
        self.model = ffvgg.get_ffvgg_model()

    def load_model(self, path):
        """
        Loads model (keras.Model instance) from the disc.
        Note that to be safe, this model should take images of (self.shape, self.shape, 3) shape
        and output self.classes number of classes.
        """
        self.model = keras.models.load_model(path)
        self.__loaded = True

    def save_model(self, path):
        """
        Save the current model.
        Please, in the path specify extension .keras or .h5
        """
        self.model.save(path)


    def load_data(self, path, plot_details=False):
        """
        Loads data from disc, splits for training and validation datasets, does some data
        preprocessing.
        In path should be be specified directory with folders named after classes and
        images in respective folders.
        """
        if self.__loaded:
            print("Pretrained model is already loaded.")
            return

        BATCH_SIZE = 32
        IMG_SIZE = (self.shape, self.shape)
        directory = path
        train_dataset = image_dataset_from_directory(directory,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE,
                                                    validation_split=0.2,
                                                    subset='training',
                                                    seed=42)
        validation_dataset = image_dataset_from_directory(directory,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,
                                                    image_size=IMG_SIZE,
                                                    validation_split=0.2,
                                                    subset='validation',
                                                    seed=42)

        self.train_dataset = train_dataset
        self.validation_dataset= validation_dataset
        
        if plot_details:
            class_names = train_dataset.class_names
            print("Classes:", class_names)

            plt.figure(figsize=(10, 10))
            for images, labels in train_dataset.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(class_names[labels[i]])
                    plt.axis("off")
            plt.show()

        # Prevents memory bottlenecks that can ocuur when reading from disc
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def train_model(self, epochs=10, batch_size=32, plot_details=False):
        """
        Train current model.
        """
        if self.__loaded:
            print("Pretrained model is already loaded.")
            return

        # For early stop
        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        history = self.model.fit(self.train_dataset,
        validation_data=self.validation_dataset,
        epochs=epochs, batch_size=batch_size)

        # Plot details
        if plot_details:
            acc = [0.] + history.history['accuracy']
            val_acc = [0.] + history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(2, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()),1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([0,12.0])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')
            plt.show()

    def predict_for_single_image(self, path, show_probs=False):
        """
        Predicts class for single input.
        """
        img = image.load_img(path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        print('Input image shape:', x.shape)
        imshow(img)
        plt.show()
        prediction = self.model.predict(x)
        if show_probs:
            for i, class_str in self.classes_dict.items():
                print(str(class_str), ":", str(prediction[0][i]))
            print()
        print("Predicted class:", self.classes_dict[np.argmax(prediction)])


if __name__ == '__main__':
    manager = ModelManager(labels=['Albedo', 'Ayaka', 'Hu Tao', 'Kokomi', 'Neither'])
    manager.create_ffvgg_model()
    manager.load_data('dataset')
    manager.train_model(epochs=80, batch_size=32, plot_details=True)
    manager.save_model('genshin5ffvggtrue.keras')