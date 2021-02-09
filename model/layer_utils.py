import tensorflow as tf
from tensorflow.keras import layers, activations
import tensorflow_addons as tfa

class ConvBlock(tf.keras.Model):
    def __init__(self, filters, k=3, s=1, p='same', nm='i'):
        """
        Args:
            nm: normalization mode. ['b','i'].(batch, instance)
        """
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(filters=filters, kernel_size=k, strides=(s,s), padding=p, use_bias=False)
        if nm=='b':
            self.norm = layers.BatchNormalization()
        elif nm=='i':
            self.norm = tfa.layers.InstanceNormalization()
        
    def call(self, x):
        out = activations.relu(self.norm(self.conv(x)))
        return out

class TransposedConvBlock(tf.keras.Model):
    def __init__(self, filters, k=3, s=2, p='same', nm='i'):
        super(TransposedConvBlock, self).__init__()
        self.conv = layers.Conv2DTranspose(filters, kernel_size=k, strides=(s,s), padding=p, use_bias=False)
        if nm=='b':
            self.norm = layers.BatchNormalization()
        elif nm=='i':
            self.norm = tfa.layers.InstanceNormalization()

    def call(self, x):
        out = activations.relu(self.norm(self.conv(x)))
        return out

class ResBlock(tf.keras.Model):
    """Modified ResBlock."""
    def __init__(self, filters, nm='i'):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(filters)
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=(1,1), padding='same', use_bias=False)
        if nm=='b':
            self.norm = layers.BatchNormalization()
        elif nm=='i':
            self.norm = tfa.layers.InstanceNormalization()

    def call(self, x):
        out = self.conv1(x)
        out = self.norm(self.conv2(x))
        return out+x



if __name__ == '__main__':
    b, h, w, c = 4, 128, 128, 3
    x = tf.random.normal((b,h,w,c))
    # blk = ConvBlock(64, s=2)
    blk = TransposedConvBlock(32, s=2)
    out = blk(x)
    blk.summary()
    print(out.shape)