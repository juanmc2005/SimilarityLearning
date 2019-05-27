import torch
import argparse
from distances import CosineDistance
from losses.base import BaseTrainer, TrainLogger, TestLogger, Evaluator, Visualizer, Optimizer, ModelSaver
from losses import config as cf
from datasets import MNIST, VoxCeleb1
from models import MNISTNet, SpeakerNet


# Constants and script arguments
loss_options = 'softmax / contrastive / triplet / arcface / center / coco'
use_cuda = torch.cuda.is_available() and True
nfeat, nclass = 2, 1251
seed = 999
device = torch.device('cuda' if use_cuda else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--mnist', type=str, default=None, help='Path to MNIST dataset')
parser.add_argument('--loss', type=str, help=loss_options)
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')
parser.add_argument('-c', '--controlled', type=bool, default=True,
                    help='Whether to set a fixed seed to control the training environment. Default value: True')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Steps (in percentage) to show epoch progress. Default value: 10')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--plot', dest='plot', action='store_true', help='Plot best accuracy dev embeddings')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='Do NOT plot best accuracy dev embeddings')
parser.set_defaults(plot=True)
parser.add_argument('--save', dest='save', action='store_true', help='Save best accuracy models')
parser.add_argument('--no-save', dest='save', action='store_false', help='Do NOT save best accuracy models')
parser.set_defaults(save=True)
parser.add_argument('--task', type=str, default='mnist', help='The task to train')


def enabled_str(value):
    return 'ENABLED' if value else 'DISABLED'


def get_config(loss):
    if loss == 'softmax':
        return cf.SoftmaxConfig(device, nfeat, nclass)
    elif loss == 'contrastive':
        return cf.ContrastiveConfig(device)
    elif loss == 'triplet':
        return cf.TripletConfig(device)
    elif loss == 'arcface':
        return cf.ArcFaceConfig(device, nfeat, nclass)
    elif loss == 'center':
        return cf.CenterConfig(device, nfeat, nclass, distance=CosineDistance())
    elif loss == 'coco':
        return cf.CocoConfig(device, nfeat, nclass)
    else:
        raise ValueError(f"Loss function should be one of: {loss_options}")


# Parse arguments and set custom seed if requested
args = parser.parse_args()
if args.controlled:
    print(f"[Seed: {seed}]")
    torch.manual_seed(seed)

# Get loss dependent configuration
config = get_config(args.loss)

# Load Dataset
print(f"[Task: {args.task}]")
print('[Loading Dataset...]')
if args.task == 'mnist' and args.mnist is not None:
    model = MNISTNet(nfeat, loss_module=config.loss_module)
    dataset = MNIST(args.mnist, args.batch_size)
elif args.task == 'speaker':
    model = SpeakerNet(nfeat, sample_rate=16000, window=200, loss_module=config.loss_module)
    dataset = VoxCeleb1(args.batch_size)
else:
    raise ValueError('Unrecognized task or MNIST path missing')
test = dataset.test_partition()
train = dataset.training_partition()
print('[Dataset Loaded]')

# Create plugins
test_callbacks = []
train_callbacks = []
if args.log_interval in range(1, 101):
    print(f"[Logging: {enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(TestLogger(args.log_interval, test.nbatches()))
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches()))
else:
    print(f"[Logging: {enabled_str(False)}]")

print(f"[Plots: {enabled_str(args.plot)}]")
if args.plot:
    test_callbacks.append(Visualizer(config.name, config.param_desc))

print(f"[Model Saving: {enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(ModelSaver(f"images/{args.loss}-best.pt"))
train_callbacks.append(Evaluator(device, test, config.test_distance, test_callbacks))

# Configure trainer
trainer = BaseTrainer(model, device, config.loss, train, config.optimizer(model, args.task), train_callbacks)

print()

# Start training
trainer.train(args.epochs)
