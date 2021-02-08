import tensorflow as tf
import sys, os
sys.path.append(os.path.abspath("model"))

def get_dataset(args):
    ...

def get_model(args):
    if args.model=='fastnst':
        from model.fastnst import FastNST
        net = FastNST()
    else:
        raise NotImplementedError()

    return net

def get_criterions(args):
    if args.model=='fastnst':
        criterion1 = tf.keras.losses.MeanSquaredError()
        criterion2 = tf.keras.losses.MeanSquaredError()
        criterions = [criterion1, criterion2]
    else:
        raise NotImplementedError()

    return criterions

def get_optimizer(args):
    if args.optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)
    else:
        raise NotImplementedError()

    return optimizer