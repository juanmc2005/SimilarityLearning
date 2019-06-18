import argparse
from common import set_custom_seed
from distances import EuclideanDistance, CosineDistance
from experiments import VoxCeleb1ModelEvaluationExperiment, SemEvalModelEvaluationExperiment

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

distance = CosineDistance() if args.distance == 'cosine' else EuclideanDistance()

print(f"[Task: {args.task.upper()}]")

print('[Preparing...]')
if args.task == 'speaker':
    experiment = VoxCeleb1ModelEvaluationExperiment(model_path=args.model,
                                                    nfeat=256,
                                                    distance=distance,
                                                    batch_size=args.batch_size)
    metric_name = 'EER'
elif args.task == 'sts':
    experiment = SemEvalModelEvaluationExperiment(model_path=args.model,
                                                  nfeat=500,
                                                  data_path=args.sts_path,
                                                  word2vec_path=args.word2vec,
                                                  vocab_path=args.vocab,
                                                  distance=distance,
                                                  log_interval=args.log_interval,
                                                  batch_size=args.batch_size,
                                                  baseline=False)
    metric_name = 'Spearman'
else:
    raise ValueError("Task can only be 'speaker' or 'sts'")

print('[Started Evaluation...]')
if args.partition == 'dev':
    metric = experiment.evaluate_on_dev()
else:
    metric = experiment.evaluate_on_test()
print(f"[Evaluation Finished]")

print(f"[{args.partition.upper()} {metric_name} = {metric}]")
