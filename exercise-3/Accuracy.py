import torch


class Accuracy:
    """A class to keep track of the accuracy while training"""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        """Resets the internal state"""
        self.correct = 0
        self.total = 0

    def update(self, output, labels):
        """
        Updates the internal state to later compute the overall accuracy

        output: the output of the network for a batch
        labels: the target labels
        """
        _, predicted = torch.max(
            output.data, 1)  # predicted now contains the predicted class index/label

        self.total += labels.size(0)
        # .item() gets the number, not the tensor
        self.correct += (predicted == labels).sum().item()

    def compute(self):
        return self.correct/self.total
