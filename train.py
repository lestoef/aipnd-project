'''
usage example:

python train.py --data_dir flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.0008 --gpu cuda
'''
import argparse
import torch
from torchvision import datasets, transforms, models
import json
from model import Classifier, train, save, load

parser = argparse.ArgumentParser(description='Train a NN model. \n Usage example: \n \
                                 python train.py --data_dir flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.0008 --gpu cuda')
parser.add_argument('--data_dir', type=str, default='flowers',
                    help='path to the training data')
parser.add_argument('--save_dir', type=str, default='checkpoints',
                    help='path to the checkpoint')
parser.add_argument('--arch', type=str, choices=['vgg16', 'densenet'], default='vgg16',
                    help='architecture of pretrained model')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate of the optimizer')
parser.add_argument('--hidden_units', type=int, default=1024,
                    help='number of nodes in the hidden layer of the classifier')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to be trained')
parser.add_argument('--gpu', type=str, default='cpu',
                    help='gpu or cpu')

args = parser.parse_args()

data_dir = args.data_dir
print(f'data_dir: {data_dir}')

save_dir = args.save_dir
print(f'save_dir: {save_dir}')

arch = args.arch
print(f'arch: {arch}')

learning_rate = args.learning_rate
print(f'learning_rate: {learning_rate}')

hidden_units = args.hidden_units
print(f'hidden_units: {hidden_units}')

epochs = args.epochs
print(f'epochs: {epochs}')

device = args.gpu
print(f'device: {device}')

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
num_classes = len(cat_to_name)

if arch == 'vgg16':
    # use VGG with 16 layers as suggested in https://arxiv.org/pdf/1409.1556.pdf
    model = models.vgg16(pretrained=True)
    # determine the size of the input layer for the classifier part
    in_features = model.classifier[0].in_features
elif arch == 'densenet':
    model = models.densenet121(pretrained=True)
    in_features = model.classifier.in_features

# switch off gradient computation for features
for param in model.parameters():
    param.requires_grad = False

# instantiate the new classifier
classifier = Classifier(in_features=in_features, hidden_units=hidden_units, out_features=num_classes)

# replace the default classifier with our custom classifier
model.classifier = classifier

print(model)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

print('Starting training')
train(device, model, train_loader, valid_loader, criterion, optimizer, epochs=epochs, print_every=32)

model.class_to_idx = train_data.class_to_idx

print('Saving model to checkpoint')
save_path = save(save_dir, arch, criterion, optimizer, epochs, learning_rate, hidden_units, model)
print(f'Saved model to {save_path}')
