import torch
import numpy as np

def train(device, model, train_loader, test_loader, criterion, optimizer, epochs=5, print_every=10):
    model.to(device)
    steps = 0
    running_loss = 0
    train_losses = []
    test_losses = []
    accuracies = []
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
                train_losses += [running_loss/print_every]
                test_losses += [valid_loss/len(test_loader)]
                accuracies += [accuracy/len(test_loader)]
                running_loss = 0
                model.train()
        # early stopping
        if (test_losses[-1] > np.mean(test_losses[-3:])):
            return train_losses, test_losses, accuracies
    return train_losses, test_losses, accuracies
