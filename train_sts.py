from os.path import join

import common
from core.base import Trainer
from core.plugins.logging import TrainLogger, TestLogger, MetricFileLogger
from core.plugins.storage import BestModelSaver, ModelLoader
from datasets.semeval import SemEval, SemEvalPartitionFactory
from sts.modes import STSForwardModeFactory
from sts.augmentation import SemEvalAugmentationStrategyFactory
from models import SemanticNet
from metrics import LogitsSpearmanMetric, DistanceSpearmanMetric, STSBaselineEvaluator, STSEmbeddingEvaluator


task = 'sts'

# Script arguments
parser = common.get_arg_parser()
parser.add_argument('--path', type=str, required=True, help='Path to SemEval dataset')
parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
parser.add_argument('--word2vec', type=str, required=True, help='Path to word embeddings')
parser.add_argument('-t', '--threshold', type=float, default=3.,
                    help='The threshold to consider a pair as positive or negative')
parser.add_argument('--remove-scores', type=list, default=[],
                    help='A list of scores to remove from the training set')
parser.add_argument('--redundancy', dest='redundancy', action='store_true', help='Allow redundancy in the training set')
parser.add_argument('--no-redundancy', dest='redundancy', action='store_false',
                    help='Do NOT allow redundancy in the training set')
parser.set_defaults(redundancy=False)
parser.add_argument('--augment', dest='augment', action='store_true', help='Augment the training set')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='Do NOT augment the training set')
parser.set_defaults(augment=False)
args = parser.parse_args()
args.remove_scores = [int(s) for s in args.remove_scores]

# Create directory to save plots, models, results, etc
log_path = common.create_log_dir(args.exp_id, task, args.loss)
print(f"Logging to {log_path}")

# Dumping all script arguments
common.dump_params(join(log_path, 'config.cfg'), args)

# Set custom seed before doing anything
common.set_custom_seed(args.seed)

# Load dataset and create model
print(f"[Task: {task.upper()}]")
print(f"[Loss: {args.loss.upper()}]")
print('[Loading Dataset...]')
print(f"[Threshold: {args.threshold}]")
print(f"[Augmentation: {common.enabled_str(args.augment)}]")
nfeat = 500
mode = STSForwardModeFactory().new(args.loss)
augmentation = SemEvalAugmentationStrategyFactory(args.loss, threshold=args.threshold,
                                                  allow_redundancy=args.redundancy,
                                                  augment=args.augment,
                                                  remove_scores=args.remove_scores)
partition_factory = SemEvalPartitionFactory(args.loss, args.batch_size)
dataset = SemEval(args.path, args.word2vec, args.vocab, augmentation.new(), partition_factory)
config = common.get_config(args.loss, nfeat, dataset.nclass, task, args.margin, args.distance,
                           args.size_average, args.loss_scale, args.triplet_strategy, args.semihard_negatives)
model = SemanticNet(common.DEVICE, nfeat, dataset.vocab, loss_module=config.loss_module, mode=mode)
dev = dataset.dev_partition()
train = dataset.training_partition()
print('[Dataset Loaded]')

# Train and evaluation plugins
test_callbacks = []
train_callbacks = []

# Logging configuration
if args.log_interval in range(1, 101):
    print(f"[Logging: {common.enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(TestLogger(args.log_interval, dev.nbatches()))
    test_callbacks.append(MetricFileLogger(log_path=join(log_path, f"metric.log")))
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches(),
                                       loss_log_path=join(log_path, f"loss.log")))
else:
    print(f"[Logging: {common.enabled_str(False)}]")

# Model saving configuration
print(f"[Model Saving: {common.enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(BestModelSaver(task, args.loss, log_path, args.exp_id))

# Evaluation configuration
if args.loss == 'kldiv':
    metric = LogitsSpearmanMetric()
    evaluator = STSBaselineEvaluator(common.DEVICE, dev, metric, test_callbacks)
else:
    metric = DistanceSpearmanMetric(config.test_distance)
    evaluator = STSEmbeddingEvaluator(common.DEVICE, dev, metric, test_callbacks)
train_callbacks.append(evaluator)

# Training configuration
trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, task, lr=(args.lr, args.loss_mod_lr)),
                  model_loader=ModelLoader(args.recover, args.recover_optim) if args.recover is not None else None,
                  callbacks=train_callbacks)
print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print()

# Start training
trainer.train(args.epochs, log_path, common.get_basic_plots(args.lr, args.batch_size, 'Spearman', 'green'))
