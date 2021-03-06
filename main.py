import comet_ml
import tensorflow as tf
import argparse
from utils import get_optimizer, get_model, train_step, \
    get_dataset, get_criterion, log_image, cycle
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
parser.add_argument("--batch-size", default=4, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--log-image-step", default=100, type=int, help="Number of steps ")
parser.add_argument("--style-coef", default=1.0, type=float)
parser.add_argument("--mixed-precision", action='store_true')
parser.add_argument("--n-mode", default='b', help="normalization mode ['b','i'] ")
args = parser.parse_args()

if args.mixed_precision:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

content_ds, style_ds = get_dataset(args)
model = get_model(args)
criterion = get_criterion(args)
optimizer = get_optimizer(args)
train_loss = tf.keras.metrics.Mean(name='train_loss')
with open("data/api_key.txt",'r') as f:
    api_key = f.readline()

logger = comet_ml.Experiment(
    api_key=api_key,
    project_name="NeuralStyleTransfer",
    auto_metric_logging=True,
    auto_param_logging=True
)
logger.set_name(f'{args.model}')
curr_step = 0
with logger.train():
    train_loss.reset_states()
    for epoch in range(1, args.epochs+1):
        for images, styles in zip(tqdm.tqdm(content_ds), cycle(style_ds)):
            if images.shape[0] != styles.shape[0]: continue
            curr_step += 1
            if curr_step%args.log_image_step==0:
                orig_images = tf.identity(images)

            loss = train_step(images, styles, model, criterion, optimizer, train_loss, args.style_coef, is_mixed=args.mixed_precision)
            logger.log_metric(name='train_loss',value=loss)
            logger.log_parameters(vars(args))
            if curr_step%args.log_image_step==0:
                log_image(orig_images, logger, curr_step, model)
        filename=f'{args.model}_transform.hdf5'
        model.trans_net.save_weights(filename)
        logger.log_asset(filename)
