import torch
import numpy as np
import argparse
from distances import CosineDistance
from losses.base import BaseTrainer, TrainLogger, TestLogger, Visualizer, ModelSaver, DeviceMapperTransform
from metrics import KNNAccuracyMetric, LogitsSpearmanMetric,\
    DistanceSpearmanMetric, STSEmbeddingEvaluator, STSBaselineEvaluator,\
    SpeakerVerificationEvaluator, ClassAccuracyEvaluator
from losses import config as cf
from datasets import MNIST, VoxCeleb1, SemEval
from models import MNISTNet, SpeakerNet, SemanticNet


# Constants and script arguments
loss_options = 'softmax / contrastive / triplet / arcface / center / coco'
use_cuda = torch.cuda.is_available() and True
seed = 124
device = torch.device('cuda' if use_cuda else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='Path to MNIST/SemEval dataset')
parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file for STS')
parser.add_argument('--word2vec', type=str, default=None, help='Path to word embeddings for STS')
parser.add_argument('--loss', type=str, help=loss_options)
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')
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
parser.add_argument('--recover', type=str, default=None, help='The path to the saved model to recover for training')


def enabled_str(value):
    return 'ENABLED' if value else 'DISABLED'


def get_config(loss, nfeat, nclass, task):
    if loss == 'softmax':
        return cf.SoftmaxConfig(device, nfeat, nclass)
    elif loss == 'contrastive':
        return cf.ContrastiveConfig(device,
                                    margin=0.15,
                                    distance=CosineDistance(),
                                    size_average=False,
                                    online=task != 'sts')
    elif loss == 'triplet':
        return cf.TripletConfig(device)
    elif loss == 'arcface':
        return cf.ArcFaceConfig(device, nfeat, nclass)
    elif loss == 'center':
        return cf.CenterConfig(device, nfeat, nclass, distance=CosineDistance())
    elif loss == 'coco':
        return cf.CocoConfig(device, nfeat, nclass)
    elif loss == 'kldiv':
        return cf.KLDivergenceConfig(device, nfeat)
    else:
        raise ValueError(f"Loss function should be one of: {loss_options}")


# Parse arguments and set custom seed
args = parser.parse_args()
print(f"[Seed: {seed}]")
torch.manual_seed(seed)
np.random.seed(seed)

# Load Dataset
print(f"[Task: {args.task.upper()}]")
print('[Loading Dataset...]')
batch_transforms = [DeviceMapperTransform(device)]
if args.task == 'mnist' and args.path is not None:
    nfeat, nclass = 2, 10
    config = get_config(args.loss, nfeat, nclass, args.task)
    model = MNISTNet(nfeat, loss_module=config.loss_module)
    dataset = MNIST(args.path, args.batch_size)
elif args.task == 'speaker':
    nfeat, nclass = 256, 1251
    config = get_config(args.loss, nfeat, nclass, args.task)
    model = SpeakerNet(nfeat, sample_rate=16000, window=200, loss_module=config.loss_module)
    dataset = VoxCeleb1(args.batch_size, segment_size_millis=200)
elif args.task == 'sts':
    nfeat = 500
    if args.loss == 'kldiv':
        mode = 'baseline'
    elif args.loss == 'contrastive':
        mode = 'pairs'
    elif args.loss == 'triplet':
        mode = 'triplets'
    else:
        mode = 'clusters'
    dataset = SemEval(args.path, args.word2vec, args.vocab, args.batch_size, mode=mode, threshold=(1.2, 3.8))
    config = get_config(args.loss, nfeat, dataset.nclass, args.task)
    model = SemanticNet(device, nfeat, dataset.vocab, loss_module=config.loss_module, mode=mode)
else:
    raise ValueError('Unrecognized task or dataset path missing')
test = dataset.dev_partition()
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
    test_callbacks.append(ModelSaver(args.task, args.loss, 'tmp'))

if args.task == 'mnist':
    evaluator = ClassAccuracyEvaluator(device, test, KNNAccuracyMetric(config.test_distance),
                                       batch_transforms, test_callbacks)
elif args.task == 'speaker':
    evaluator = SpeakerVerificationEvaluator(device, args.batch_size, config.test_distance,
                                             dataset.config, test_callbacks)
# STS
elif args.loss == 'kldiv':
    metric = LogitsSpearmanMetric()
    evaluator = STSBaselineEvaluator(device, test, metric, batch_transforms, test_callbacks)
else:
    metric = DistanceSpearmanMetric(config.test_distance)
    evaluator = STSEmbeddingEvaluator(device, test, metric, batch_transforms, test_callbacks)

train_callbacks.append(evaluator)

# Configure trainer
trainer = BaseTrainer(args.loss, model, device, config.loss, train, config.optimizer(model, args.task),
                      recover=args.recover, batch_transforms=batch_transforms, callbacks=train_callbacks)

print()

# Start training
trainer.train(args.epochs)
