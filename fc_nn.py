import torch
import numpy as np
from pprint import pprint


class NN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


def main():
    x = np.random.rand(100, 1)  # input data
    # y = 2 * x + 3  # output data
    y = np.sqrt(2 * x**2 + 4)  # output function to approximate

    inputs = torch.from_numpy(x).float()
    labels = torch.from_numpy(y).float()

    model = NN(1, 10, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                'Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 1000, loss.item()))

    # Test the model on the data set on which it was originally trained on
    pprint(model(inputs)[:5])
    pprint(labels[:5])


if __name__ == '__main__':
    main()
