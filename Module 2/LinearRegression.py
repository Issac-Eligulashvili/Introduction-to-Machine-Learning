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
W = torch.randn(784, 10) / np.sqrt(784) #Generated from the 28x28 size grid. Generate a random weight in a matrix sized 784x10 then scaled down for variance
W.requires_grad_() # Let torch know to track its gradient
b = torch.zeros(10, requires_grad=True) # Make a blank bias vector with dimensions 10x1

# Optimizer
optimizer = torch.optim.SGD([W, b], lr=0.1) # Calculate the stochastic gradient descent for W and b with a learning rate of .1

# Iterate through train set minibatches
for images, labels in tqdm(train_loader):
    # Zero out the gradients
    optimizer.zero_grad()

    # Forward pass
    x = images.view(-1, 28 * 28) # Reshape images
    y = torch.matmul(x, W) + b # Compute the probabilities
    cross_entropy = F.cross_entropy(y, labels) #Calculate loss
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
        x = images.view(-1, 28 * 28)
        y = torch.matmul(x, W) + b

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct / total))