import argparse
from utils import set_custom_seed, DEVICE
from distances import EuclideanDistance, CosineDistance
from experiments import VoxCeleb1ModelEvaluationExperiment


# Script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--distance', type=str, default='euclidean', help='cosine / euclidean. Default: euclidean')
parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training and testing')
parser.add_argument('--model', type=str, default=None, help='The path to the saved model to evaluate')
args = parser.parse_args()

# Set custom seed
set_custom_seed()

print('[Preparing...]')
nfeat, nclass = 256, 1251
distance = CosineDistance() if args.distance == 'cosine' else EuclideanDistance()
experiment = VoxCeleb1ModelEvaluationExperiment(model_path=args.model,
                                                nfeat=nfeat,
                                                distance=distance,
                                                device=DEVICE,
                                                batch_size=args.batch_size)
print('[Started Evaluation...]')
eer = experiment.evaluate_on_test()
print(f"[Evaluation Finished. TEST EER = {eer}]")
