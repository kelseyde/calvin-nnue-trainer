import time as t

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from fen import fen_to_features
from src import util


class EPDFileDataset(Dataset):
    def __init__(self, file_path, max_size=None, delimiter=','):
        self.file_path = file_path
        self.max_size = max_size
        self.delimiter = delimiter
        self.length = 0
        self.index_offsets = []
        self._build_index()

    def _build_index(self):
        offset = 0
        with open(self.file_path, 'r') as f:
            for idx, line in enumerate(f):
                self.index_offsets.append(offset)
                offset += len(line)
                if self.max_size is not None and idx >= self.max_size - 1:
                    break
        self.length = len(self.index_offsets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as f:
            f.seek(self.index_offsets[idx])
            line = f.readline()
            return parse_labelled_position(line, self.delimiter)


def load(file_path, batch_size=64, max_size=None, delimiter=','):
    start = t.time()

    dataset = EPDFileDataset(file_path, max_size, delimiter)
    num_data = len(dataset)
    train_size = int(0.9 * num_data)
    val_size = num_data - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    end = t.time()
    print(f"data loaded in {end - start:.2f}s")
    print(f"data size: {num_data}, ")

    return train_loader, val_loader


def parse_labelled_position(line, delimiter=','):
    parts = line.split(delimiter)
    fen_string = parts[0].strip()
    input_data = fen_to_features(fen_string).astype(np.float32)  # Convert to float32
    result = parse_result(parts[1])
    score = util.cp_to_wdl(parts[2])
    output_data = None
    if result is not None and score is not None:
        output_data = (np.float32(result), np.float32(score))
    elif result is not None:
        output_data = np.float32(result)
    elif score is not None:
        output_data = np.float32(score)
    return input_data, output_data


def parse_result(result):
    if '1-0' in result or "1.0" in result:
        return 1.0
    elif '0-1' in result or "0.0" in result:
        return 0.0
    elif '1/2-1/2' in result or "0.5" in result:
        return 0.5
    else:
        return None