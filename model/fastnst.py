"""
Implementation for Fast Neural Transformation Net[Johnson, J.(ECCV'16)].
This architecture follows https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
"""

import tensorflow as tf
from tensorflow.keras import layers, activations
import sys, os
sys.path.append(os.path.abspath("model"))
from layer_utils import ConvBlock, ResBlock, TransposedConvBlock

class TransformationNet(tf.keras.Model):
    def __init__(self):
        super(TransformationNet, self).__init__()
        self.enc1 = ConvBlock(32, k=9, s=1)
        self.enc2 = ConvBlock(64, k=3, s=2)
        self.enc3 = ConvBlock(128, k=3, s=2)

        self.blc1 = ResBlock(128)
        self.blc2 = ResBlock(128)
        self.blc3 = ResBlock(128)
        self.blc4 = ResBlock(128)
        self.blc5 = ResBlock(128)

        self.dec1 = TransposedConvBlock(64, k=3, s=2)
        self.dec2 = TransposedConvBlock(32, k=3, s=2)
        self.last = ConvBlock(3, k=9, s=1)

    def call(self, x):
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.blc1(out)
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.last(out)
        return out

class FastNST(tf.keras.Model):
    def __init__(self):
        super(FastNST, self).__init__()
        self.trans_net = TransformationNet()
        base = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        self.loss_net = tf.keras.Model(base.input, base.layers[-6].output)
        self.loss_net.trainable = False
        self.outputs_name = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    def call(self, x, training=True):
        out = self.trans_net(x, training=training)
        outputs = self.extract(out, training=training)
        return outputs

    def transform(self, x, training=False):
        """feed-forward for transformation net"""
        out = self.trans_net(x, training=training)
        return out

    def extract(self, x, training=False):
        """feed-forward for loss net"""
        outputs = []
        for layer in self.loss_net.layers:
            x = layer(x, training=training)
            if layer.name in self.outputs_name:
                outputs.append(x)

        return outputs


if __name__ == '__main__':
    b,h,w,c = 4, 224, 224, 3
    x = tf.random.normal((b,h,w,c))
    # net = TransformationNet()
    # import IPython; IPython.embed(); exit(1)
    net = FastNST()
    net.build(input_shape=(None,h,w,c))
    net.summary()
    outputs = net(x)
    gen = net.transform(x)
    print(len(outputs))
    print(gen.shape)