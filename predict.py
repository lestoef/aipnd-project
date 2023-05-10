from model import load
from utils import process_image, predict, show_probs
import argparse

parser = argparse.ArgumentParser(description='Predict a class. \n Usage example: \n \
                                 python predict.py --sample_path flowers')
parser.add_argument('--sample_path', type=str, default='flowers/test/4/image_05658.jpg',
                    help='path to the sample')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/vgg16-NLLLoss-torch.optim.adam-epochs-3.pth',
                    help='path to the checkpoint')
parser.add_argument('--top_k', type=int, default=5,
                    help='top k probabilities')
parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                    help='.json file containing categories and associated names')
parser.add_argument('--gpu',
                    help='toggle gpu')

args = parser.parse_args()

sample_path = args.sample_path
print(f'sample_path: {sample_path}')

checkpoint_path = args.checkpoint_path
print(f'checkpoint_path: {checkpoint_path}')

top_k = args.top_k
category_names = args.category_names

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'

model, optimizer, criterion, learning_rate, epoch = load(checkpoint_path, device)

probs, classes = predict(sample_path, model, top_k)
print(probs)
print(classes)

img = process_image(sample_path)
show_probs(img, probs, classes, cat_to_name_json=category_names)