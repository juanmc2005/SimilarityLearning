from os.path import join

import common
from core.base import Trainer
from core.plugins.logging import TrainLogger, TestLogger, MetricFileLogger, HeaderPrinter
from core.plugins.storage import BestModelSaver, ModelLoader
from datasets.semeval import SemEval, SemEvalPartitionFactory
from sts.modes import STSForwardModeFactory
from sts.augmentation import SemEvalAugmentationStrategyFactory
from models import SemanticNet
from metrics import LogitsSpearmanMetric, DistanceSpearmanMetric, STSBaselineEvaluator, STSEmbeddingEvaluator
from gensim.models import Word2Vec


task = 'sts'

# Script arguments
parser = common.get_arg_parser()
parser.add_argument('--path', type=str, required=True, help='Path to SemEval dataset')
parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
parser.add_argument('--word2vec', type=str, required=False, default=None, help='Path to word embeddings')
parser.add_argument('--word2vec-model', type=str, required=False, default=None, help='Path to GENSIM Word2Vec model')
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
vocab_vec = dataset.vocab_vec if args.word2vec is not None else Word2Vec.load(args.word2vec_model).wv
model = SemanticNet(common.DEVICE, nfeat, 1, dataset.vocab, vocab_vec, loss_module=config.loss_module, mode=mode)
dev = dataset.dev_partition()
test = dataset.test_partition()
train = dataset.training_partition()
print('[Dataset Loaded]')

# Train and evaluation plugins
test_callbacks = []
train_callbacks = [HeaderPrinter()]

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
    dev_evaluator = STSBaselineEvaluator(common.DEVICE, dev, metric, 'dev', test_callbacks)
    test_evaluator = STSBaselineEvaluator(common.DEVICE, test, metric, 'test',
                                          callbacks=[MetricFileLogger(log_path=join(log_path, 'test-metric.log'))])
else:
    metric = DistanceSpearmanMetric(config.test_distance)
    dev_evaluator = STSEmbeddingEvaluator(common.DEVICE, dev, metric, 'dev', test_callbacks)
    test_evaluator = STSEmbeddingEvaluator(common.DEVICE, test, metric, 'test',
                                           callbacks=[MetricFileLogger(log_path=join(log_path, 'test-metric.log'))])
train_callbacks.extend([dev_evaluator, test_evaluator])

# Training configuration
trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, task, lr=(args.lr, args.loss_mod_lr)),
                  model_loader=ModelLoader(args.recover, args.recover_optim) if args.recover is not None else None,
                  callbacks=train_callbacks, last_metric_fn=lambda: dev_evaluator.last_metric)
print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print(f"[Model Size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.1f}m]")
print()

# Start training
trainer.train(args.epochs, log_path, common.get_basic_plots(args.lr, args.batch_size, 'Spearman', 'green'))
