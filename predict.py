from model import load
from utils import process_image, predict, show_probs
import argparse

parser = argparse.ArgumentParser(description='Predict a class. \n Usage example: \n \
                                 python predict.py --sample_path flowers')
parser.add_argument('--sample_path', type=str, default='flowers/test/4/image_05658.jpg',
                    help='path to the sample')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/vgg16-NLLLoss-torch.optim.adam-epochs-3.pth',
                    help='path to the checkpoint')

args = parser.parse_args()

sample_path = args.sample_path
print(f'sample_path: {sample_path}')

checkpoint_path = args.checkpoint_path
print(f'checkpoint_path: {checkpoint_path}')

device = 'cpu'

model, optimizer, criterion, learning_rate, epoch = load(checkpoint_path, device)

probs, classes = predict(sample_path, model)
print(probs)
print(classes)

img = process_image(sample_path)
show_probs(img, probs, classes)