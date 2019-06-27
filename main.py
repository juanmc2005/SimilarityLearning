import argparse
import time
from os.path import join

from core.base import Trainer
from core.plugins.logging import TrainLogger, TestLogger, MetricFileLogger
from core.plugins.visual import Visualizer, SpeakerDistanceVisualizer
from core.plugins.storage import BestModelSaver, ModelLoader, RegularModelSaver
from core.plugins.misc import TrainingMetricCalculator

from datasets.mnist import MNIST
from datasets.semeval import SemEval, SemEvalPartitionFactory
from datasets.voxceleb import VoxCeleb1

from metrics import KNNAccuracyMetric, LogitsSpearmanMetric, LogitsAccuracyMetric, \
    DistanceSpearmanMetric, STSEmbeddingEvaluator, STSBaselineEvaluator, \
    SpeakerVerificationEvaluator, ClassAccuracyEvaluator
from models import MNISTNet, SpeakerNet, SemanticNet
from sts.modes import STSForwardModeFactory
from sts.augmentation import SemEvalAugmentationStrategyFactory
from common import LOSS_OPTIONS_STR, get_config, enabled_str, set_custom_seed, DEVICE, create_log_dir

launch_datetime = time.strftime('%c')

# Script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None, help='Path to MNIST/SemEval dataset')
parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file for STS')
parser.add_argument('--word2vec', type=str, default=None, help='Path to word embeddings for STS')
parser.add_argument('--loss', type=str, help=LOSS_OPTIONS_STR)
parser.add_argument('--epochs', type=int, help='The number of epochs to run the model')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Steps (in percentage) to show epoch progress. Default value: 10')
parser.add_argument('--eval-interval', type=int, default=10,
                    help='Steps (in epochs) to evaluate the speaker model. Default value: 10')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--plot', dest='plot', action='store_true', help='Plot best accuracy dev embeddings')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='Do NOT plot best accuracy dev embeddings')
parser.set_defaults(plot=True)
parser.add_argument('--save', dest='save', action='store_true', help='Save best accuracy models')
parser.add_argument('--no-save', dest='save', action='store_false', help='Do NOT save best accuracy models')
parser.set_defaults(save=True)
parser.add_argument('--task', type=str, default='mnist', help='The task to train')
parser.add_argument('--recover', type=str, default=None, help='The path to the saved model to recover for training')
parser.add_argument('-m', '--margin', type=float, default=2., help='The margin to use for the losses that need it')
parser.add_argument('-t', '--threshold', type=float, default=3.,
                    help='The threshold for STS to consider a pair positive or negative')
parser.add_argument('--remove-scores', type=list, default=[],
                    help='A list of scores to remove from the STS training set')
parser.add_argument('--redundancy', dest='redundancy', action='store_true',
                    help='Allow redundancy in SemEval training set')
parser.add_argument('--no-redundancy', dest='redundancy', action='store_false',
                    help='Do NOT allow redundancy in SemEval training set')
parser.set_defaults(redundancy=False)
parser.add_argument('--exp-id', type=str, default=f"EXP-{launch_datetime.replace(' ', '-')}",
                    help='An identifier for the experience')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

args.remove_scores = [int(s) for s in args.remove_scores]

log_path = create_log_dir(args.exp_id, args.task, args.loss)
print(f"Logging to {log_path}")

# Set custom seed before doing anything
set_custom_seed(args.seed)

# Load Dataset
print(f"[Task: {args.task.upper()}]")
print(f"[Loss: {args.loss.upper()}]")
print('[Loading Dataset...]')
if args.task == 'mnist' and args.path is not None:
    nfeat, nclass = 2, 10
    config = get_config(args.loss, nfeat, nclass, args.task, args.margin)
    model = MNISTNet(nfeat, loss_module=config.loss_module)
    dataset = MNIST(args.path, args.batch_size)
elif args.task == 'speaker':
    dataset = VoxCeleb1(args.batch_size, segment_size_millis=200)
    train = dataset.training_partition()
    nfeat, nclass = 256, train.nclass
    config = get_config(args.loss, nfeat, nclass, args.task, args.margin)
    model = SpeakerNet(nfeat, sample_rate=16000, window=200, loss_module=config.loss_module)
    print(f"Train Classes: {nclass}")
    print(f"[Batches per Epoch: {train.batches_per_epoch}]")
elif args.task == 'sts' and args.path is not None:
    print(f"[Threshold: {args.threshold}]")
    nfeat = 500
    mode = STSForwardModeFactory().new(args.loss)
    augmentation = SemEvalAugmentationStrategyFactory(args.loss, threshold=args.threshold,
                                                      allow_redundancy=args.redundancy,
                                                      remove_scores=args.remove_scores)
    partition_factory = SemEvalPartitionFactory(args.loss, args.batch_size)
    dataset = SemEval(args.path, args.word2vec, args.vocab, augmentation.new(), partition_factory)
    config = get_config(args.loss, nfeat, dataset.nclass, args.task, args.margin)
    model = SemanticNet(DEVICE, nfeat, dataset.vocab, loss_module=config.loss_module, mode=mode)
else:
    raise ValueError('Unrecognized task or dataset path missing')

dev = dataset.dev_partition()
train = dataset.training_partition()

print('[Dataset Loaded]')

# Create plugins
test_callbacks = []
train_callbacks = []
if args.log_interval in range(1, 101):
    print(f"[Logging: {enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(TestLogger(args.log_interval, dev.nbatches()))
    test_callbacks.append(MetricFileLogger(log_path=join(log_path, f"metric.log")))
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches(),
                                       loss_log_path=join(log_path, f"loss.log")))
else:
    print(f"[Logging: {enabled_str(False)}]")

print(f"[Plots: {enabled_str(args.plot)}]")
if args.plot:
    test_callbacks.append(Visualizer(config.name, config.param_desc))

print(f"[Model Saving: {enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(BestModelSaver(args.task, args.loss, log_path, args.exp_id))

if args.task == 'mnist':
    evaluator = ClassAccuracyEvaluator(DEVICE, dev, KNNAccuracyMetric(config.test_distance), test_callbacks)
elif args.task == 'speaker':
    train_callbacks.extend([TrainingMetricCalculator(name='Training Accuracy',
                                                     metric=LogitsAccuracyMetric(),
                                                     file_path=join(log_path, 'train-accuracy.log')),
                            RegularModelSaver(args.task, args.loss, log_path, interval=5, experience_name=args.exp_id)])
    test_callbacks.append(SpeakerDistanceVisualizer(log_path))
    evaluator = SpeakerVerificationEvaluator(args.batch_size, config.test_distance,
                                             args.eval_interval, dataset.config, test_callbacks)
# STS
elif args.loss == 'kldiv':
    metric = LogitsSpearmanMetric()
    evaluator = STSBaselineEvaluator(DEVICE, dev, metric, test_callbacks)
else:
    metric = DistanceSpearmanMetric(config.test_distance)
    evaluator = STSEmbeddingEvaluator(DEVICE, dev, metric, test_callbacks)

train_callbacks.append(evaluator)

# Configure trainer
trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, args.task, args.lr),
                  model_loader=ModelLoader(args.recover) if args.recover is not None else None,
                  callbacks=train_callbacks)

print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print()

# Start training
trainer.train(args.epochs)
