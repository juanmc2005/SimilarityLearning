from os.path import join

import common
from core.base import Trainer
from core.plugins.logging import TrainLogger, MetricFileLogger, HeaderPrinter
from core.plugins.storage import BestModelSaver, RegularModelSaver, ModelLoader
from core.plugins.visual import SpeakerDistanceVisualizer
from core.plugins.misc import TrainingMetricCalculator
from datasets.voxceleb import VoxCeleb1
from models import SpeakerNet
from metrics import LogitsAccuracyMetric, SpeakerVerificationEvaluator


task = 'speaker'

# Script arguments
parser = common.get_arg_parser()
parser.add_argument('--eval-interval', type=int, default=10,
                    help='Steps (in epochs) to evaluate the speaker model. Default value: 10')
parser.add_argument('--save-interval', type=int, default=10,
                    help='Steps (in epochs) to save the speaker model. Default value: 10')
parser.add_argument('--triplet-strategy', type=str, default='all',
                    help=F'Triplet sampling strategy. Possible values: {common.TRIPLET_SAMPLING_OPTIONS_STR}')
parser.add_argument('--semihard-negatives', type=int, default=10,
                    help='The number of negatives to keep when using a semi-hard negative triplet sampling strategy')
parser.add_argument('--segments-per-speaker', type=int, default=1,
                    help='The number of audio segments per speaker in each training batch')
parser.add_argument('--segment-length', type=int, default=200,
                    help='The length of each audio segment in milliseconds. Default: 200ms')
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
print(f"[Loss: {args.loss.upper()}]")
print('[Loading Dataset...]')
nfeat = 256
dataset = VoxCeleb1(args.batch_size,
                    segment_size_millis=args.segment_length,
                    segments_per_speaker=args.segments_per_speaker)
train = dataset.training_partition()
config = common.get_config(args.loss, nfeat, train.nclass, task, args.margin,
                           args.triplet_strategy, args.semihard_negatives)
model = SpeakerNet(nfeat, sample_rate=16000, window=args.segment_length, loss_module=config.loss_module)
print(f"[Train Classes: {train.nclass}]")
print(f"[Batches per Epoch: {train.batches_per_epoch}]")
print('[Dataset Loaded]')

# Train and evaluation plugins
test_callbacks: list = []
train_callbacks: list = [HeaderPrinter()]

# Logging configuration
if args.log_interval in range(1, 101):
    print(f"[Logging: {common.enabled_str(True)} (every {args.log_interval}%)]")
    test_callbacks.append(MetricFileLogger(log_path=join(log_path, 'metric.log')))
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches(),
                                       loss_log_path=join(log_path, 'loss.log')))
else:
    print(f"[Logging: {common.enabled_str(False)}]")

# Model saving configuration
print(f"[Model Saving: {common.enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(BestModelSaver(task, args.loss, log_path, args.exp_id))

# Plotting configuration
print(f"[Plotting: {common.enabled_str(args.plot)}]")
if args.plot:
    test_callbacks.append(SpeakerDistanceVisualizer(log_path))
    plots = common.get_basic_plots(args.lr, args.batch_size, 'EER', 'red')
    if args.loss not in ['triplet', 'contrastive']:
        plots.append({
            'log_file': 'train-accuracy.log',
            'metric': 'Accuracy',
            'color': 'green',
            'title': f'Train Accuracy - lr={args.lr} - batch_size={args.batch_size}',
            'filename': 'train-accuracy-plot'
        })
else:
    plots = []

# Other useful plugins
if args.loss not in ['triplet', 'contrastive']:
    train_callbacks.append(TrainingMetricCalculator(name='Training Accuracy',
                                                    metric=LogitsAccuracyMetric(),
                                                    file_path=join(log_path, 'train-accuracy.log')))

train_callbacks.append(RegularModelSaver(task, args.loss, log_path,
                                         interval=args.save_interval,
                                         experience_name=args.exp_id))

# Evaluation configuration
evaluators = [SpeakerVerificationEvaluator('development', args.batch_size, config.test_distance,
                                           args.eval_interval, dataset.config, test_callbacks),
              SpeakerVerificationEvaluator('test', args.batch_size, config.test_distance,
                                           args.eval_interval, dataset.config,
                                           callbacks=[MetricFileLogger(log_path=join(log_path, 'test-metric.log'))])]
train_callbacks.extend(evaluators)

# Training configuration
trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, task, args.lr),
                  model_loader=ModelLoader(args.recover) if args.recover is not None else None,
                  callbacks=train_callbacks)
print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print()

# Start training
trainer.train(args.epochs, log_path, plots)
