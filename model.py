import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

class Classifier(nn.Module):
    def __init__(self, in_features=1024, hidden_units=1024, out_features=2):
        super(Classifier, self).__init__()
        self.seq = nn.Sequential(OrderedDict([
                          ('fc1', torch.nn.Linear(in_features=in_features, out_features=hidden_units, bias=True)),
                          ('relu1', torch.nn.ReLU(inplace=True)),
                          ('dropout1', torch.nn.Dropout(p=0.5, inplace=False)),
                          ('fc2', torch.nn.Linear(in_features=hidden_units, out_features=out_features, bias=True)),
                          ('output', torch.nn.LogSoftmax(dim=1))
                          ])
        )

    def forward(self, x):
        return self.seq(x)
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def train(device, model, train_loader, test_loader, criterion, optimizer, epochs=5, print_every=10):
    model.to(device)
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}, "
                    f"Train loss: {running_loss/print_every:.3f}, "
                    f"Test loss: {valid_loss/len(test_loader):.3f} "
                    f"Accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()

def save(save_dir, arch, criterion, optimizer, epochs, learning_rate, hidden_units, model):
    save_path = f'{save_dir}/{arch}-{criterion._get_name()}-{optimizer.__module__}-epochs-{epochs}.pth'

    torch.save({
                'arch': arch,
                'epoch': epochs,
                'criterion': criterion._get_name(),
                'optimizer': optimizer.__module__,
                'learning_rate': learning_rate,
                'hidden_units': hidden_units,
                'class_to_index': model.class_to_idx,
                'classifier_state_dict': model.classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
    return save_path

def load(save_path, device):
    checkpoint = torch.load(save_path, map_location=device)
    if checkpoint['arch']=='densenet':
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
    elif checkpoint['arch']=='vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
    else:
        raise ValueError
    # switch off gradient computation for features
    for param in model.parameters():
        param.requires_grad = False
    num_classes = len(checkpoint['class_to_index'])
    # instantiate the new classifier
    model.classifier = Classifier(in_features=in_features, out_features=num_classes)
    if checkpoint['criterion']=='NLLLoss':
        criterion = torch.nn.NLLLoss()
    else:
        raise ValueError
    learning_rate = checkpoint['learning_rate']
    if checkpoint['optimizer']=='torch.optim.adam':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
    model.class_to_idx = checkpoint['class_to_index']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, criterion, learning_rate, epoch