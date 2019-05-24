import numpy as np
from pathlib import Path
import nltk
import re
from torch.utils.data import Dataset
import torch

# nltk.download('punkt')


class ImdbDataset(Dataset):
    def __init__(self, input_dir: str):
        super().__init__()
        self.neg_path = Path(input_dir + '/neg')
        self.pos_path = Path(input_dir + '/pos')
        self.files_neg = list(self.neg_path.iterdir())
        self.files_pos = list(self.pos_path.iterdir())
        self.num_neg = len(self.files_neg)
        self.num_pos = len(self.files_pos)

        self.word_vec = Path(
            input_dir + '/../imdb.vocab').read_text().split('\n')

    def __len__(self):
        return len(self.files_neg) + len(self.files_pos)

    def __getitem__(self, idx: int):
        if idx < self.num_neg:
            path = Path(self.files_neg[idx])
            review_label = torch.tensor(0).long()
        else:
            path = Path(self.files_pos[idx - self.num_neg])
            review_label = torch.tensor(1).long()
        review_txt = path.read_text()
        review_words = nltk.word_tokenize(
            re.sub(r"\(|\)|\,|\.|\?|\!|<br />", "", review_txt))

        review_input = np.zeros(len(self.word_vec))
        for elem in review_words:
            try:
                pos = self.word_vec.index(elem)
                review_input[pos] = 1
            except:
                pass

        return torch.from_numpy(review_input).float(), review_label
