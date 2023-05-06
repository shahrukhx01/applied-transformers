from torch.utils.data import Dataset
import pandas as pd

class TransformerDataset(Dataset):
    def __init__(self, csv_file_path, src_column, tgt_column):
        """
        Arguments:
            csv_file_path (string): Path to the csv file
        """
        dataset = pd.read_csv(csv_file_path)
        self.src_texts = dataset[src_column].values
        self.tgt_texts = dataset[tgt_column].values

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.tgt_texts[idx]
