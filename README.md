# Image classification via Convolutional Neural Networks

## A set up
The module may be used via python interpreter

```python
>>> from operation_module import *
```

## Create a new model

I'll use a Genshin Impact characters dataset (https://www.kaggle.com/datasets/just1ce5/genshin-impact-characters-dataset). While creating an instance, it's recommended to give labels in the appropriate (usually, alphabetical) order as a parameter.
```python
>>> manager = ModelManager(labels=['Albedo', 'Ayaka', 'Hu Tao', 'Kokomi', 'Neither'])
>>> manager = create_residual_model()
```
The code above creates a ResNet50.

## Load data

```python
>>> manager.load_data('dataset/', plot_details=True)
...
Classes: ['Albedo', 'Ayaka', 'Hu Tao', 'Kokomi', 'Neither']
```
Here are some images from dataset plotted.
![](https://github.com/just1ce415/convnet_classifier/blob/main/images/data_plot2.jpg)

## Train and evaluate model
```python
>>> manager.train_model(epochs=80, batch_size=32, plot_details=True)
```
It may take some time...

Here are plotted details on training/validation loss/accuracy.
![](https://github.com/just1ce415/convnet_classifier/blob/main/images/train_plot.jpg)

## Save trained model

```python
>>> manager.save_model('genshin5resnet.keras')
```
In the future, it could be loaded
```python
>>> manager.load_model('genshin5resnet.keras')
```

## Other models
Using ResNet50 for building model for prediction of only 5 classes may be an overkill (which could be seen by graphs above - training is very slow). Now we can use simpler architecture FFVGG-13 (basically, VGG-13 with vanilla convolutions replaced by Fast Fourier convolutions; FFC appeared to work slower than optimized Keras Conv2D, so we replaced FFC by them until we optimize it) implemented.

```python
>>> manager2 = ModelManager(labels=['Albedo', 'Ayaka', 'Hu Tao', 'Kokomi', 'Neither'])
>>> manager2 = create_ffvgg_model()
>>> manager2.train_model(epochs=80, batch_size=32, plot_details=True)
>>> manager2.save_model('genshin5ffvgg.keras')
```

Output metrics:
![](https://github.com/just1ce415/convnet_classifier/blob/main/images/train_plot2.jpg)

Looks much better now. By the graphs of validation accuracy and loss we can conclude that there was an overfit after ~65th epoch. Also, our dataset needs improvement.

## Predict for single image
The resulting model easily classifies obvious images as well as images from training set.
```python
>>> manager2.predict_for_single_image('single_tests\\ayaya.jpg', show_probs=True)
```
There will be input an image plot:
![](https://github.com/just1ce415/convnet_classifier/blob/main/images/image_pred_plot.jpg)

And also probabilities shown for each class:
```python
Input image shape: (1, 256, 256, 3)
Albedo : 4.4303197e-31
Ayaka : 0.9945639
Hu Tao : 9.747406e-14
Kokomi : 3.809505e-05
Neither : 0.0053980392

Predicted class: Ayaka
```
However, it may still do some mistakes when situaltion is not clear. E.g. it may classify Lumin (Neither) as Albedo considering that they have similarity.

Lumin:

![](https://github.com/just1ce415/convnet_classifier/blob/main/images/lumin.jpg)

Albedo:

![](https://github.com/just1ce415/convnet_classifier/blob/main/images/albedo.jpg)

Impovements and new architectures coming soon.

## The End :)
