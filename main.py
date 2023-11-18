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
    minlimit = 0
    maxlimit = 15
    # Prepare data
    X = torch.arange(minlimit, maxlimit, 0.001).unsqueeze(dim=1)
    Y = torch.sin(X) + torch.cos(X)
    
    device = 'cuda:0'
    #device = 'cpu'
    X = X.to(device)
    Y = Y.to(device)

    # Define dataset
    dataset = TensorDataset(X, Y)

    # Define data loader
    train_loader = DataLoader(dataset, batch_size=500, shuffle=True)

    # Initialize model
    hidden_size = 48
    model = Net(hidden_size)
    
    # Load model 
    model.load_state_dict(torch.load(f'models/sin_prediction_{hidden_size}.pt'))
    
    model = model.to(device)
    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=learning_rate)
    loss_fn = nn.MSELoss()

    print(model.state_dict())

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        print()
        plt.clf()
        for batch ,(x , y) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)

            # Compute the loss
            loss : torch.Tensor = loss_fn(outputs, y)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step() 

            
            print(f"Epoch {epoch+1}, Batch {batch+1}/{len(train_loader)}, Loss {loss.item()}", end='\r')
            
            # Update the plot           
            plt.plot(x.cpu().numpy(), y.cpu().numpy() ,'b.')
            plt.plot(x.cpu().numpy(), outputs.detach().cpu().numpy(), 'r.')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.ylim((-1.5, 1.5))
            plt.xlim((-0.5 + minlimit, 0.5 + maxlimit))
            plt.title('Original Data vs. Predicted Data')
            plt.pause(0.001)
    
    
    with torch.inference_mode() : 
        model.eval()
        outputs = model(X)
        plt.plot(X.cpu().numpy(), outputs.detach().cpu().numpy(), 'y.')
        plt.show()
    
    torch.save(model.state_dict(), f'models/sin_prediction_{hidden_size}.pt')

if __name__ == "__main__":
    main()
