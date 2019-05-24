import torch.nn as nn


class ImdbModel(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()

        self.input_size = input_size
        self.class_num = class_num
        self.fc1_size = 1024

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.fc1_size, self.class_num)

        self.initWeights()

    def forward(self, x):
        o1 = self.relu1(self.fc1(x))
        o2 = self.fc2(o1)
        return o2

    def initWeights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
