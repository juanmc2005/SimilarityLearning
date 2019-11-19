import argparse
import time
from os.path import join
from torch import nn, optim
import common
from metrics import LogitsAccuracyMetric, STSBaselineEvaluator
from sts.augmentation import SNLINoNeutralAugmentation
from sts.modes import ConcatSTSForwardMode
from datasets.snli import SNLI
from models import SemanticNet, SNLIClassifierNet
from core.base import Trainer
from core.optim import Optimizer
from core.plugins.logging import TrainLogger, TestLogger, MetricFileLogger
from core.plugins.storage import BestModelSaver, ModelLoader
from losses.wrappers import LossWrapper

launch_datetime = time.strftime('%c')

# Script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, required=True, help='The number of epochs to run the model')
parser.add_argument('--model', type=str, required=True, help='The path to the saved model to evaluate')
parser.add_argument('--distance', type=str, default='euclidean', help='cosine / euclidean. Default: euclidean')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--path', type=str, default=None, help='Path to SNLI dataset')
parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file for STS')
parser.add_argument('--word2vec', type=str, default=None, help='Path to word embeddings for STS')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Steps (in percentage) to show evaluation progress, only for STS. Default: 10')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--exp-id', type=str, default=f"EXP-{launch_datetime.replace(' ', '-')}",
                    help='An identifier for the experience')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--save', dest='save', action='store_true', help='Save best accuracy models')
parser.add_argument('--no-save', dest='save', action='store_false', help='Do NOT save best accuracy models')
parser.set_defaults(save=True)
args = parser.parse_args()

# Create the directory for logs
log_dir = common.create_log_dir(args.exp_id, 'snli', 'clf-train')

# Set custom seed
common.set_custom_seed(args.seed)

distance = common.to_distance_object(args.distance)

print("[Task: SNLI]")

print('[Preparing...]')
model_loader = ModelLoader(args.model, restore_optimizer=False)
nfeat = 4096

# The augmentation is only done for the training set, so it doesn't matter which one we choose.
label2int = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
augmentation = SNLINoNeutralAugmentation(label2int)
dataset = SNLI(args.path, args.word2vec, args.vocab, args.batch_size, augmentation, label2int)

# TODO Using DEV to train as a temporary measure to test if this approach works. WILL CHANGE LATER
train_and_dev = dataset.dev_partition()
test = dataset.test_partition()

classifier = SNLIClassifierNet(common.DEVICE, model_loader,
                               nfeat_sent=nfeat, nclass=len(label2int.keys()),
                               nlayers=1, vector_vocab=dataset.vocab)

# Train and evaluation plugins
test_callbacks = []
train_callbacks = []

# Logging configuration
if args.log_interval in range(1, 101):
    print(f"[Logging: {common.enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(TestLogger(args.log_interval, train_and_dev.nbatches()))
    test_callbacks.append(MetricFileLogger(log_path=join(log_dir, f"metric.log")))
    train_callbacks.append(TrainLogger(args.log_interval, train_and_dev.nbatches(),
                                       loss_log_path=join(log_dir, f"loss.log")))
else:
    print(f"[Logging: {common.enabled_str(False)}]")

# Model saving configuration
print(f"[Model Saving: {common.enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(BestModelSaver('snli', 'softmax', log_dir, args.exp_id))

# Evaluation configuration
# TODO evaluate on different partition than train !!!
evaluators = [STSBaselineEvaluator(common.DEVICE, train_and_dev, LogitsAccuracyMetric(),
                                   partition_name='train',
                                   callbacks=test_callbacks),
              STSBaselineEvaluator(common.DEVICE, test, LogitsAccuracyMetric(),
                                   partition_name='test',
                                   callbacks=[TestLogger(args.log_interval, test.nbatches()),
                                              MetricFileLogger(log_path=join(log_dir, f"test-metric.log"))])]
train_callbacks.extend(evaluators)

trainer = Trainer('softmax', classifier, LossWrapper(nn.NLLLoss().to(common.DEVICE)), train_and_dev,
                  Optimizer([optim.RMSprop(classifier.parameters(), lr=args.lr)], []),
                  callbacks=train_callbacks)

print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print(f"[Model Size: {sum(p.numel() for p in classifier.parameters() if p.requires_grad) / 1000000:.1f}m]")
print()

# Start training
trainer.train(args.epochs, log_dir, common.get_basic_plots(args.lr, args.batch_size, 'Accuracy', 'green'))
