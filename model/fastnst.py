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
        self.loss_net = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        self.loss_net.trainable = False

    def call(self, x):
        out = self.trans_net(x)
        out = self.loss_net(out)
        return out

    def transform(self, x):
        out = self.trans_net(x)
        return out


if __name__ == '__main__':
    b,h,w,c = 4, 224, 224, 3
    x = tf.random.normal((b,h,w,c))
    # net = TransformationNet()
    net = FastNST()
    net.build(input_shape=(None,h,w,c))
    net.summary()
    # import IPython; IPython.embed(); exit(1)
    out = net(x)
    gen = net.transform(x)
    print(out.shape)
    print(gen.shape)