import argparse
import time
from common import set_custom_seed, create_log_dir, to_distance_object
from core.plugins.storage import ModelLoader
from core.plugins.visual import DetCurveVisualizer, SpeakerDistanceVisualizer
from experiments.semeval import SemEvalEmbeddingEvaluationExperiment, SemEvalBaselineModelEvaluationExperiment
from experiments.voxceleb import VoxCeleb1ModelEvaluationExperiment

launch_datetime = time.strftime('%c')

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
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--exp-id', type=str, default=f"EXP-{launch_datetime.replace(' ', '-')}",
                    help='An identifier for the experience')
args = parser.parse_args()

# Create the directory for logs
log_dir = create_log_dir(args.exp_id, args.task, 'eval')

# Set custom seed
set_custom_seed(args.seed)

distance = to_distance_object(args.distance)

print(f"[Task: {args.task.upper()}]")

print('[Preparing...]')
if args.task == 'speaker':
    experiment = VoxCeleb1ModelEvaluationExperiment(model_path=args.model,
                                                    nfeat=256,
                                                    distance=distance,
                                                    batch_size=args.batch_size,
                                                    verification_callbacks=[SpeakerDistanceVisualizer(log_dir),
                                                                            DetCurveVisualizer(log_dir)])
    metric_name = 'EER'
elif args.task == 'sts':
    model_loader = ModelLoader(args.model, restore_optimizer=False)
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
                                 batch_size=args.batch_size,
                                 base_dir='tmp')
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
