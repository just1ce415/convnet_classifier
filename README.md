# Image classification via Convolutional Neural Networks

## A set up
The module may be used via python interpreter

```python
>>> from operation_module import *
```

## Create a new model

I'll use a Genshin Impact characters dataset (API command: kaggle datasets download -d just1ce5/genshin-impact-characters-dataset). While creating instance, it's recommended to give labels in the appropriate (usually, alphabetical) order as a parameter.
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

## Predict for single image
```python
>>> manager.predict_for_single_image('single_tests\\ayaya.jpg', show_probs=True)
```
There will be input an image plot:
![](https://github.com/just1ce415/convnet_classifier/blob/main/images/image_pred_plot.jpg)

And also probabilities shown for each class:
```python
Input image shape: (1, 256, 256, 3)
Albedo : 0.0018577169
Ayaka : 0.0008746913
Hu Tao : 0.78035784
Kokomi : 0.012803782
Neither : 0.20410599

Predicted class: Hu Tao
```

As you see from here and plots above, there's still some problem with the model...

## The End :)