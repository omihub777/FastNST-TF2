import comet_ml
import tensorflow as tf
import argparse
from utils import get_optimizer, get_model, train_step, get_dataset, get_criterion, image_grid
import itertools
import math
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='fastnst', type=str)
parser.add_argument("--content-path",  default="data/content", type=str)
parser.add_argument("--style-path", default="data/trainB",type=str)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--learning-rate", default=1e-3, type=float)
parser.add_argument("--beta-1", default=0.9, type=float)
parser.add_argument("--beta-2", default=0.999, type=float)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--log-image-step", default=100, type=int, help="Number of steps ")
parser.add_argument("--style-coef", default=1.0, type=float)
args = parser.parse_args()

with open("data/api_key.txt",'r') as f:
    api_key = f.readline()

content_ds, style_ds = get_dataset(args)
model = get_model(args)
criterion = get_criterion(args)
optimizer = get_optimizer(args)
train_loss = tf.keras.metrics.Mean(name='train_loss')

logger = comet_ml.Experiment(
    api_key=api_key,
    project_name="NeuralStyleTransfer",
    auto_metric_logging=True,
    auto_param_logging=True
)
logger.set_name(f'{args.model}')
num_images = args.batch_size
with logger.train():
    train_loss.reset_states()
    for epoch in range(1, args.epochs+1):
        for step, (images, styles) in enumerate(zip(tqdm.tqdm(content_ds), itertools.cycle(style_ds)), 1):
            curr_step = epoch*step
            if curr_step%args.log_image_step==0:
                images = images[:num_images]
                orig_images = tf.identity(images[:num_images])



            loss = train_step(images, styles, model, criterion, optimizer, train_loss, args.style_coef)
            logger.log_metric(name='train_loss',value=loss)
            logger.log_parameters(vars(args))
            if curr_step%args.log_image_step==0:
                tf.stop_gradient(orig_images)
                orig_grids = image_grid(orig_images, size=int(math.sqrt(num_images)))
                logger.log_image(orig_grids, step=curr_step)
                gen_images = model.transform(orig_images)
                gen_grids = image_grid(gen_images, size=int(math.sqrt(num_images)))
                logger.log_image(gen_grids, step=curr_step)

        filename=f'{args.model}_transform.hdf5'
        model.trans_net.save_weights(filename)
        logger.log_asset(filename)
