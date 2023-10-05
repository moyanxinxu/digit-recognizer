import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class myDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.labels = self.data["label"].to_numpy()
        self.features = self.data.drop("label", axis=1).to_numpy()

        self.features = self.features.reshape((-1, 1, 28, 28))
        # self.labels = self.labels.reshape((-1, 1))

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index])
        return feature, label

    def __len__(self):
        return len(self.data)


def load_data(file_path, batch_size):
    data = myDataSet(file_path)
    train, test = random_split(data, [0.8, 0.2])  # TODO
    train_iter = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
    test_iter = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter
