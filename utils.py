import tensorflow as tf
import sys, os
sys.path.append(os.path.abspath("model"))
import glob

def get_image_paths(data_path, fmt="png"):
    filenames = glob.glob(f"{data_path}/*.{fmt}")
    return filenames

def get_dataset(args):
    """Returns (images, styles)"""
    def parse_func(filename):
        p = tf.io.read_file(filename)
        img = tf.io.decode_png(p, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (args.size, args.size))
        return img

    content_filenames, style_filenames = get_image_paths(args.content_path), get_image_paths(args.style_path)
    content_ds = tf.data.Dataset.from_tensor_slices(content_filenames)
    content_ds = content_ds.shuffle(len(content_filenames))
    content_ds = content_ds.map(parse_func)
    content_ds = content_ds.batch(args.batch_size)
    content_ds = content_ds.prefetch(1)

    style_ds = tf.data.Dataset.from_tensor_slices(style_filenames)
    style_ds = style_ds.shuffle(len(style_filenames))
    style_ds = style_ds.map(parse_func)
    style_ds = style_ds.batch(args.batch_size)
    style_ds = style_ds.prefetch(1)

    return content_ds, style_ds

def get_model(args):
    if args.model=='fastnst':
        from model.fastnst import FastNST
        net = FastNST()
    else:
        raise NotImplementedError()

    return net

def get_criterions(args):
    if args.model=='fastnst':
        criterion = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError()

    return criterion

def get_optimizer(args):
    if args.optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)
    else:
        raise NotImplementedError()

    return optimizer

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_step(images, styles, model, criterion, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        s_outputs = model.loss_net(styles, trainig=True)
        c_output = model.loss_net(images, trainig=True)[2]

        content_loss = criterion(output[2], c_output)
        style_loss = tf.reduce_sum([criterion(gram_matrix(output), gram_matrix(s_output)) for output, s_output in zip(outputs, s_outputs)])
        loss = content_loss+style_loss
    gradients = tape.gradien(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

