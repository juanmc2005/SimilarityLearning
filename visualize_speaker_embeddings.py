import argparse
from distances import CosineDistance, EuclideanDistance
from experiments.voxceleb import VoxCeleb1TSNEVisualizationExperiment

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='The path to the saved model to use')
parser.add_argument('--distance', type=str, required=True,
                    help='The distance to calculate t-SNE. Possible values: cosine/euclidean')
parser.add_argument('-o', '--out', type=str, required=True, help='The name of the output plot file (without extension)')
args = parser.parse_args()

if args.distance == 'cosine':
    distance = CosineDistance()
elif args.distance == 'euclidean':
    distance = EuclideanDistance()
else:
    raise ValueError('Distance parameter should be either cosine or euclidean')

print('[Preparing...]')
experiment = VoxCeleb1TSNEVisualizationExperiment(args.model, 256, distance)

print('[Experiment started]')
experiment.visualize_dev(50, 'tmp', args.out)
print('Done')
