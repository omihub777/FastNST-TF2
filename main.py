import comet_ml
import tensorflow as tf
import argparse
from utils import get_optimzier, get_model


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='fastnst', type=str)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--learning-rate", default=1e-3, type=float)
parser.add_argument("--beta-1", default=0.9, type=float)
parser.add_argument("--beta-2", default=0.999, type=float)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--epochs", default=10, type=int)
args = parser.parse_args()


model = get_model(args)
criterions = get_criterion(args)
optimizer = get_optimzier(args)


