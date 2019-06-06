import torch
import argparse
import numpy as np
from datasets import SemEval
from models import SemanticNet
from metrics import SpearmanMetric
import losses.config as cf
from losses.base import TrainLogger, TestLogger, ModelSaver, Evaluator, BaseTrainer, DeviceMapperTransform


def enabled_str(value):
    return 'ENABLED' if value else 'DISABLED'


use_cuda = torch.cuda.is_available() and True
seed = 999
device = torch.device('cuda' if use_cuda else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='Path to SemEval dataset')
parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file')
parser.add_argument('--word2vec', type=str, default=None, help='Path to word embeddings')
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Steps (in percentage) to show epoch progress. Default value: 10')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--save', dest='save', action='store_true', help='Save best accuracy models')
parser.add_argument('--no-save', dest='save', action='store_false', help='Do NOT save best accuracy models')
parser.set_defaults(save=True)
parser.add_argument('--recover', type=str, default=None, help='The path to the saved model to recover for training')

# Parse arguments and set custom seed
args = parser.parse_args()
print(f"[Seed: {seed}]")
torch.manual_seed(seed)
np.random.seed(seed)

print('[Task: STS]')
print('[Loading Dataset...]')
nfeat = 500
dataset = SemEval(args.path, args.word2vec, args.vocab, args.batch_size)
config = cf.KLDivergenceConfig(device, nfeat)
model = SemanticNet(device, nfeat, dataset.vocab, pairwise=True, loss_module=config.loss_module)
metric = SpearmanMetric()
test = dataset.test_partition()
train = dataset.training_partition()
print('[Dataset Loaded]')

# Create plugins
test_callbacks = []
train_callbacks = []
batch_transforms = [DeviceMapperTransform(device)]
if args.log_interval in range(1, 101):
    print(f"[Logging: {enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(TestLogger(args.log_interval, test.nbatches()))
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches()))
else:
    print(f"[Logging: {enabled_str(False)}]")

print(f"[Model Saving: {enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(ModelSaver('kldiv', f"images/sts-kldiv-best.pt"))
train_callbacks.append(Evaluator(device, test, metric, batch_transforms=batch_transforms, callbacks=test_callbacks))

# Configure trainer
trainer = BaseTrainer('kldiv', model, device, config.loss, train, config.optimizer(model, 'sts'),
                      recover=args.recover, batch_transforms=batch_transforms, callbacks=train_callbacks)

print()

# Start training
trainer.train(args.epochs)
