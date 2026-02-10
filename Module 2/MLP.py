import numpy as np #Import numpy
import torch #Import pytorch
import torch.nn.functional as F #Import the softmax function
from torchvision import datasets, transforms #Import the datasets
from tqdm import tqdm #Import the progress bar

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

## Training
# Initialize parameters
W1 = torch.randn(784, 500) / np.sqrt(784) # Generate a random weight in a matrix sized 784x500 then scaled down for variance. These are the weights for the hidden layer
W1.requires_grad_() # Let torch know to track its gradient
b1 = torch.zeros(500, requires_grad=True) # Make a blank bias vector with dimensions 500x1 for the hidden layer

W2 = torch.randn(500, 10) / np.sqrt(500) #Generated from the 28x28 size grid. Generate a random weight in a matrix sized 500x10 then scaled down for variance. These are weights for output layer
W2.requires_grad_() # Let torch know to track its gradient
b2 = torch.zeros(10, requires_grad=True) # Make a blank bias vector with dimensions 10x1 for the output layer

epochs = 10

# Optimizer
optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=0.1) # Calculate the stochastic gradient descent for W and b with a learning rate of .1



# Iterate through train set minibatches
for _ in range(epochs):
    for images, labels in tqdm(train_loader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images.view(-1, 28 * 28)  # Reshape images

        hidden = torch.matmul(x, W1) + b1  # Calculate hidden layer
        hidden = F.relu(hidden)  # Add non-linearity so that it doesnt collapse to a linear  regression.
        y = torch.matmul(hidden, W2) + b2  # Calculate output layer

        cross_entropy = F.cross_entropy(y, labels)  # Calculate loss
        # Backward pass
        cross_entropy.backward()
        optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatches
    for images, labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28 * 28)  # Reshape images

        hidden = torch.matmul(x, W1) + b1  # Calculate hidden layer
        hidden = F.relu(hidden)  # Add non-linearity so that it doesnt collapse to a linear  regression.
        y = torch.matmul(hidden, W2) + b2  # Calculate output layer

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct / total))