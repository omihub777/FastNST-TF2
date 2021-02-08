import comet_ml
import tensorflow as tf
import argparse
from utils import get_optimzier, get_model, train_step, get_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='fastnst', type=str)
parser.add_argument("--api-key", required=True)
parser.add_argument("--content-path", required=True)
parser.add_argument("--style-path", required=True)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--learning-rate", default=1e-3, type=float)
parser.add_argument("--beta-1", default=0.9, type=float)
parser.add_argument("--beta-2", default=0.999, type=float)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--epochs", default=10, type=int)
args = parser.parse_args()


train_ds = get_dataset(args)
model = get_model(args)
criterion = get_criterion(args)
optimizer = get_optimzier(args)

logger = comet_ml.Experiment(
    api_key=args.api_key,
    project_name="NeuralStyleTransfer",
    auto_metric_logging=True,
    auto_param_logging=True
)
logger.set_name(f'{args.model}')
with logger.train():
    for epoch in range(args.epochs):
        for images, styles in train_ds:
            train_step(images, styles, model, criterion, optimizer)
            gen_images = model.transform(images)
            logger.log_image(gen_images)
    filename=f'{args.model}_transform.hdf5'
    model.trans_net.save_weights(filename)
    logger.log_asset(filename)
