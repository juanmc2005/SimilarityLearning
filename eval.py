import argparse
import time
from common import set_custom_seed, create_log_dir
from core.plugins.storage import ModelLoader
from experiments.semeval import SemEvalEmbeddingEvaluationExperiment, SemEvalBaselineModelEvaluationExperiment
from experiments.sst import BinarySSTClassicEvaluationExperiment
from losses.arcface import ArcLinear
from losses.center import CenterLinear
from losses.coco import CocoLinear
from distances import Distance

launch_datetime = time.strftime('%c')

# Script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='sts / sst2')
parser.add_argument('--embedding-size', type=int, required=False, default=500, help='Embedding size')
parser.add_argument('--model', type=str, required=True, help='The path to the saved model to evaluate')
parser.add_argument('--partition', type=str, required=True, help='dev / test')
parser.add_argument('--distance', type=str, default='euclidean', help='cosine / euclidean. Default: euclidean')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--path', type=str, default=None, help='Path to dataset if needed')
parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file for STS')
parser.add_argument('--word2vec', type=str, required=False, default=None, help='Path to word embeddings')
parser.add_argument('--word2vec-model', type=str, required=False, default=None, help='Path to GENSIM Word2Vec model')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Steps (in percentage) to show evaluation progress, only for STS. Default: 10')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--exp-id', type=str, default=f"EXP-{launch_datetime.replace(' ', '-')}",
                    help='An identifier for the experience')
args = parser.parse_args()

# Create the directory for logs
log_dir = create_log_dir(args.exp_id, args.task, 'eval')

# Set custom seed
set_custom_seed(args.seed)

distance = Distance.from_name(args.distance)

print(f"[Task: {args.task.upper()}]")

print('[Preparing...]')
if args.task == 'sts':
    model_loader = ModelLoader(args.model, restore_optimizer=False)
    loss_name = model_loader.get_trained_loss()
    if loss_name == 'kldiv':
        experiment_type = SemEvalBaselineModelEvaluationExperiment
    else:
        experiment_type = SemEvalEmbeddingEvaluationExperiment
    experiment = experiment_type(model_loader=model_loader,
                                 nfeat=args.embedding_size,
                                 data_path=args.path,
                                 word2vec_path=args.word2vec,
                                 word2vec_model_path=args.word2vec_model,
                                 vocab_path=args.vocab,
                                 distance=distance,
                                 log_interval=args.log_interval,
                                 batch_size=args.batch_size,
                                 base_dir='tmp')
    metric_name = 'Spearman'
elif args.task == 'sst2':
    # FIXME this only tests using logits, NOT with embedding distance
    model_loader = ModelLoader(args.model, restore_optimizer=False)
    loss_name = model_loader.get_trained_loss()
    if loss_name == 'arcface':
        loss_module = ArcLinear(args.embedding_size, 2, margin=.1, s=7.)
    elif loss_name == 'coco':
        loss_module = CocoLinear(args.embedding_size, 2, alpha=15.)
    elif loss_name in ['center', 'softmax']:
        loss_module = CenterLinear(args.embedding_size, 2)
    else:
        loss_module = None
    experiment = BinarySSTClassicEvaluationExperiment(model_loader=model_loader,
                                                      nfeat=args.embedding_size,
                                                      data_path=args.path,
                                                      vocab_path=args.vocab,
                                                      word2vec_model_path=args.word2vec_model,
                                                      distance=distance,
                                                      log_interval=args.log_interval,
                                                      batch_size=args.batch_size,
                                                      base_dir='tmp',
                                                      loss_module=loss_module)
    metric_name = 'Loss Module Accuracy'
else:
    raise ValueError("Task can only be 'speaker', 'sts' or 'sst2'")

print('[Started Evaluation...]')
if args.partition == 'dev':
    metric = experiment.evaluate_on_dev(True)
elif args.partition == 'test':
    metric = experiment.evaluate_on_test()
else:
    raise ValueError('Partition can only be: dev / test')
print(f"[Evaluation Finished]")

print(f"[{args.partition.upper()} {metric_name} = {metric}]")
