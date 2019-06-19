import argparse
from common import set_custom_seed
from losses.base import ModelLoader
from distances import EuclideanDistance, CosineDistance
from experiments.semeval import SemEvalEmbeddingEvaluationExperiment, SemEvalBaselineModelEvaluationExperiment
from experiments.voxceleb import VoxCeleb1ModelEvaluationExperiment

# Script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, help='speaker / sts')
parser.add_argument('--model', type=str, required=True, help='The path to the saved model to evaluate')
parser.add_argument('--partition', type=str, required=True, help='dev / test')
parser.add_argument('--distance', type=str, default='euclidean', help='cosine / euclidean. Default: euclidean')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--sts-path', type=str, default=None, help='Path to SemEval dataset')
parser.add_argument('--vocab', type=str, default=None, help='Path to vocabulary file for STS')
parser.add_argument('--word2vec', type=str, default=None, help='Path to word embeddings for STS')
parser.add_argument('--log-interval', type=int, default=10,
                    help='Steps (in percentage) to show evaluation progress, only for STS. Default: 10')
args = parser.parse_args()

# Set custom seed
set_custom_seed()

if args.distance == 'cosine':
    distance = CosineDistance()
elif args.distance == 'euclidean':
    distance = EuclideanDistance()
else:
    raise ValueError("Distance can only be: cosine / euclidean")

print(f"[Task: {args.task.upper()}]")

print('[Preparing...]')
if args.task == 'speaker':
    experiment = VoxCeleb1ModelEvaluationExperiment(model_path=args.model,
                                                    nfeat=256,
                                                    distance=distance,
                                                    batch_size=args.batch_size)
    metric_name = 'EER'
elif args.task == 'sts':
    model_loader = ModelLoader(args.model)
    loss_name = model_loader.get_trained_loss()
    if loss_name == 'kldiv':
        experiment_type = SemEvalBaselineModelEvaluationExperiment
    else:
        experiment_type = SemEvalEmbeddingEvaluationExperiment
    experiment = experiment_type(model_loader=model_loader,
                                 nfeat=500,
                                 data_path=args.sts_path,
                                 word2vec_path=args.word2vec,
                                 vocab_path=args.vocab,
                                 distance=distance,
                                 log_interval=args.log_interval,
                                 batch_size=args.batch_size)
    metric_name = 'Spearman'
else:
    raise ValueError("Task can only be 'speaker' or 'sts'")

print('[Started Evaluation...]')
if args.partition == 'dev':
    metric = experiment.evaluate_on_dev(True)
elif args.partition == 'test':
    metric = experiment.evaluate_on_test()
else:
    raise ValueError('Partition can only be: dev / test')
print(f"[Evaluation Finished]")

print(f"[{args.partition.upper()} {metric_name} = {metric}]")
