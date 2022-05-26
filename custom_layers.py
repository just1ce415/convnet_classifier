import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input
from keras.models import Model

class FFC2D(tf.keras.layers.Layer):
    """
    Implementation of Fast Fouries convlution.
    """
    def __init__(self, nkernels, kernel_size, **kwargs):
        self.nkernels = nkernels
        self.kernel_size= kernel_size
        super(FFC2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=self.kernel_size + (input_shape[-1], self.nkernels),
                            initializer=glorot_uniform,
                            trainable=True)
        self.bias = self.add_weight(shape=(self.nkernels,),
                            initializer=glorot_uniform,
                            trainable=True)
        super(FFC2D, self).build(input_shape)

    def call(self, x):
        crop_size = self.kernel.get_shape().as_list()[0] // 2
        shape = x.get_shape().as_list()[1] + self.kernel.get_shape().as_list()[0] - 1
        X = tf.transpose(x, perm=[0,3,1,2])
        W = tf.transpose(self.kernel, perm=[3,2,0,1])
        X = tf.signal.rfft2d(X, [shape, shape])
        W = tf.signal.rfft2d(W, [shape, shape])
        X = tf.einsum('imkl,jmkl->ijkl', X, W)
        output = tf.signal.irfft2d(X, [shape, shape])
        output = tf.transpose(output, perm=[0,2,3,1])
        output = tf.nn.bias_add(output, self.bias)[:,crop_size:-1*crop_size, crop_size:-1*crop_size, :]
        return output   

    def get_config(self):
        config = super().get_config()
        config.update({
            "nkernels": self.nkernels,
            "kernel_size": self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == '__main__':
    X_input = Input((32, 32, 3))
    X = FFC2D(16, (3,3))(X_input)
    model = Model(inputs=X_input, outputs=X)
    print(model.summary())