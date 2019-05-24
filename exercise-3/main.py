from ImdbDataset import ImdbDataset
from ImdbModel import ImdbModel
from Accuracy import Accuracy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

TRAIN_DATA = './imdb-data/train'
VALIDATION_DATA = './imdb-data/validation'
TEST_DATA = './imdb-data/test'

torch.manual_seed(123)
np.random.seed(123)

train_loader = DataLoader(ImdbDataset(TRAIN_DATA),
                          batch_size=512, shuffle=True, num_workers=49)
validation_loader = DataLoader(ImdbDataset(VALIDATION_DATA),
                               batch_size=256, shuffle=True, num_workers=5)
model = ImdbModel(89527, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
accuracy = Accuracy()

model.train()
for epoch in range(5):
    accuracy.reset()

    print("Starting epoch {}".format(epoch+1))
    total = 0
    running_loss = 0.0

    loader = tqdm(enumerate(train_loader), total=len(train_loader))
    # print(len(data_loader))

    for i, data in loader:

        inputs, labels = data
        # print("test")
        # print(len(inputs))

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss
        total += logits.size(0)

        loader.set_description("loss: {:.5f}".format(running_loss/total))
        accuracy.update(logits, labels)

    print("Accuracy: {:.2f}%".format(100 * accuracy.compute()))

accuracy.reset()

model.train(False)
with torch.no_grad():
    for data in tqdm(validation_loader):
        inputs, labels = data
        outputs = model(inputs)
        accuracy.update(outputs, labels)

print("Accuracy: {:.2f}%".format(100 * accuracy.compute()))


# print(inputs)
# print(labels)
