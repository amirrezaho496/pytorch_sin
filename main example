import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Define your model here
class Net(nn.Module):
    def __init__(self, hidden_size : int = 1024):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(1,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1)
        )
        

    def forward(self, x):
        x = self.fc1(x)
        return x

# Define your dataset here
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Define your main function here
def main():
    # Prepare data
    X = torch.arange(0, 30, 0.01).unsqueeze(dim=1)
    Y = torch.sin(X) + torch.cos(X)
    
    device = 'cuda:0'
    #device = 'cpu'
    X = X.to(device)
    Y = Y.to(device)

    # Define dataset
    dataset = TensorDataset(X.__reversed__(), Y.__reversed__())

    # Define data loader
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model and optimizer
    model = Net(hidden_size= 512).to(device)
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        print()
        for batch ,(x , y) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()

            #print(x,y)
            # Forward pass
            outputs = model(x)

            # Compute the loss
            loss : torch.Tensor = loss_fn(outputs, y)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step() 

            
            print(f"Epoch {epoch+1}, Batch {batch}/{len(train_loader)}, Loss {loss.item()}", end='\r')

        with torch.inference_mode() :
            model.eval()
            preds = model(X)
            
            # Update the plot
            plt.clf()
            plt.plot(X.cpu().numpy(), Y.cpu().numpy(), 'b.')
            plt.plot(X.cpu().numpy(), preds.cpu().numpy(), 'r.')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim((-1.5, 1.5))
            plt.xlim((-0.5, 30.5))
            plt.title('Original Data vs. Predicted Data')
            plt.pause(0.01)
    
    plt.show()

if __name__ == "__main__":
    main()
