from os.path import join

import common
from core.base import Trainer
from core.plugins.logging import TrainLogger, MetricFileLogger, HeaderPrinter
from core.plugins.storage import BestModelSaver, ModelLoader
from core.plugins.misc import IntraClassDistanceStatLogger
from datasets.ami import AMI
from aminet import AMILSTM, AMIBert
from models import MetricNet
from metrics import KNNF1ScoreMetric, ClassAccuracyEvaluator
from gensim.models import Word2Vec
from transformers import BertTokenizer


task = 'ami'

# Script arguments
parser = common.get_arg_parser()
parser.add_argument('--path', type=str, required=True, help='Path to AMI dataset')
parser.add_argument('--model', type=str, required=True, help='Model to train: lstm / bert')
parser.add_argument('--vocab', type=str, required=False, help='Path to vocabulary file')
parser.add_argument('--word2vec-model', type=str, required=False, default=None, help='Path to GENSIM Word2Vec model')
args = parser.parse_args()

# Create directory to save plots, models, results, etc
log_path = common.create_log_dir(args.exp_id, task, args.loss)
print(f"Logging to {log_path}")

# Dumping all script arguments
common.dump_params(join(log_path, 'config.cfg'), args)

# Set custom seed before doing anything
common.set_custom_seed(args.seed)

# Load dataset and create model
print(f"[Task: {task.upper()}]")
print(f"[Model: {args.model.upper()}]")
print(f"[Loss: {args.loss.upper()}]")
print('[Loading Dataset and Model...]')

# Embedding dim is 768 based on BERT, we use the same for LSTM to be fair
# Classes are 6 because we include a 'non-misogyny class'
nfeat, nclass = 768, 6
dataset = AMI(args.path)
config = common.get_config(args.loss, nfeat, nclass, task,
                           args.margin, args.distance,
                           args.size_average, args.loss_scale,
                           args.triplet_strategy, args.semihard_negatives)

if args.model == 'lstm':
    vocab_vec = Word2Vec.load(args.word2vec_model).wv
    vocab = [line.strip() for line in open(args.vocab, 'r')]
    encoder = AMILSTM(nfeat_word=300, nfeat_sent=nfeat,
                      word_list=vocab, vec_vocab=vocab_vec, dropout=0.2)
    loader = dataset.word2vec_loader(args.batch_size, balance_train=False)
elif args.model == 'bert':
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    encoder = AMIBert(pretrained_weights, freeze=False)
    loader = dataset.bert_loader(tokenizer, args.batch_size, balance_train=False)
else:
    raise ValueError(f"Unknown model '{args.model}'. Only 'lstm' and 'bert' are accepted")

model = MetricNet(encoder=encoder, classifier=config.loss_module)
dev = loader.dev_partition()
test = loader.test_partition()
train = loader.training_partition()
print('[Dataset and Model Loaded]')

# Train and evaluation plugins
test_callbacks = []
train_callbacks: list = [HeaderPrinter()]

# Logging configuration
if args.log_interval in range(1, 101):
    print(f"[Logging: {common.enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(MetricFileLogger(log_path=join(log_path, f"metric.log")))
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches(),
                                       loss_log_path=join(log_path, f"loss.log")))
    test_callbacks.append(IntraClassDistanceStatLogger(config.test_distance, join(log_path, 'mean-dists.log')))
else:
    print(f"[Logging: {common.enabled_str(False)}]")

# Model saving configuration
print(f"[Model Saving: {common.enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(BestModelSaver(task, args.loss, log_path, args.exp_id))

# Evaluation configuration
metric = KNNF1ScoreMetric(config.test_distance, neighbors=10)
dev_evaluator = ClassAccuracyEvaluator(common.DEVICE, dev, metric, 'dev', test_callbacks)
test_evaluator = ClassAccuracyEvaluator(common.DEVICE, test, metric, 'test',
                                        callbacks=[MetricFileLogger(log_path=join(log_path, 'test-metric.log'))])
train_callbacks.extend([dev_evaluator, test_evaluator])

# Training configuration
trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, task, lr=(args.lr, args.clf_lr)),
                  model_loader=ModelLoader(args.recover, args.recover_optim) if args.recover is not None else None,
                  callbacks=train_callbacks, last_metric_fn=lambda: dev_evaluator.last_metric)
print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print(f"[Model Size: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.1f}m]")
print()

# Start training
trainer.train(args.epochs, log_path, common.get_basic_plots(args.lr, args.batch_size, 'Macro F1', 'green'))

print(f"Best result at epoch {dev_evaluator.best_epoch}:")
print(f"Dev: {dev_evaluator.best_metric}")
print(f"Test: {test_evaluator.results[dev_evaluator.best_epoch-1]}")
