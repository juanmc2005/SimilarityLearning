from os.path import join

import common
from core.base import Trainer
from core.plugins.logging import TrainLogger, TestLogger, MetricFileLogger
from core.plugins.storage import BestModelSaver, ModelLoader
from core.plugins.visual import Visualizer
from datasets.mnist import MNIST
from models import MNISTNet
from metrics import KNNAccuracyMetric, ClassAccuracyEvaluator


task = 'mnist'

# Script arguments
parser = common.get_arg_parser()
parser.add_argument('--path', type=str, required=True, help='Path to MNIST dataset')
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
nfeat, nclass = 2, 10
config = common.get_config(args.loss, nfeat, nclass, task, args.margin)
model = MNISTNet(nfeat, loss_module=config.loss_module)
dataset = MNIST(args.path, args.batch_size)

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

# Plotting configuration
print(f"[Plots: {common.enabled_str(args.plot)}]")
if args.plot:
    test_callbacks.append(Visualizer(config.name, config.param_desc))

# Model saving configuration
print(f"[Model Saving: {common.enabled_str(args.save)}]")
if args.save:
    test_callbacks.append(BestModelSaver(task, args.loss, log_path, args.exp_id))

# Evaluation configuration
metric = KNNAccuracyMetric(config.test_distance)
train_callbacks.append(ClassAccuracyEvaluator(common.DEVICE, dev, metric, test_callbacks))

# Training configuration
trainer = Trainer(args.loss, model, config.loss, train, config.optimizer(model, task, args.lr),
                  model_loader=ModelLoader(args.recover) if args.recover is not None else None,
                  callbacks=train_callbacks)
print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print()

# Start training
trainer.train(args.epochs, log_path, common.get_basic_plots(args.lr, args.batch_size, 'Accuracy', 'green'))
