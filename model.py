import torch
from torch import nn
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