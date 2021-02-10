import tensorflow as tf
import sys, os
sys.path.append(os.path.abspath("model"))
import glob
import math

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

    content_filenames, style_filenames = get_image_paths(args.content_path), get_image_paths(args.style_path, fmt='jpg')
    # import IPython; IPython.embed(); exit(1)
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
        net = FastNST(n_mode=args.n_mode)
    else:
        raise NotImplementedError()

    return net

def get_criterion(args):
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

def gram_matrix(input_tensor, is_mixed):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    if is_mixed:
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float16)
    else:
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def calc_loss(criterion, outputs, c_output, s_outputs, style_coef, is_mixed):
    content_loss = criterion(outputs[2], c_output)
    style_loss = tf.reduce_sum([criterion(gram_matrix(output, is_mixed), gram_matrix(s_output, is_mixed)) for output, s_output in zip(outputs, s_outputs)])
    loss = content_loss+ style_coef * style_loss
    return loss


@tf.function
def train_step(images, styles, model, criterion, optimizer,train_loss, style_coef=1., is_mixed=False):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        c_output = model.extract(images)[2]
        s_outputs = model.extract(styles)

        # import IPython; IPython.embed(); exit(1)
        loss = calc_loss(criterion, outputs, c_output, s_outputs, style_coef, is_mixed)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss = train_loss(loss)
    return loss

def image_grid(x, size=6):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image


# @tf.function
def log_image(orig_images,logger, curr_step, model): 
    num_images = orig_images.shape[0]
    orig_grids = image_grid(orig_images, size=int(math.sqrt(num_images)))
    logger.log_image(orig_grids, step=curr_step)
    gen_images = model.transform(tf.stop_gradient(orig_images))
    gen_grids = image_grid(gen_images, size=int(math.sqrt(num_images)))
    logger.log_image(gen_grids, step=curr_step)

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)