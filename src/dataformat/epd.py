import time as t

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.dataformat.fen import fen_to_features
from src import wdl


class EPDFileDataset(Dataset):
    def __init__(self, file_path, max_size=None, delimiter=',', fen_index=0, result_index=1, score_index=2):
        self.file_path = file_path
        self.max_size = max_size
        self.delimiter = delimiter
        self.length = 0
        self.index_offsets = []
        self.fen_index = fen_index
        self.result_index = result_index
        self.score_index = score_index
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
            return parse_labelled_position(line, self.delimiter, self.fen_index, self.result_index, self.score_index)


def load(file_path, batch_size=64, max_size=None, device="mps", delimiter=',', fen_index=0, result_index=1, score_index=2):
    start = t.time()

    dataset = EPDFileDataset(file_path, max_size, delimiter, fen_index, result_index, score_index)
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


def parse_labelled_position(line, delimiter=',', fen_index=0, result_index=1, score_index=2, device="mps"):
    parts = line.split(delimiter)
    fen_string = parts[fen_index].strip()
    stm_features, nstm_features, stm = fen_to_features(fen_string)
    result = parse_result(parts[result_index], stm)
    score = parse_score(parts[score_index], stm)
    if result is None and score is not None:
        result = score
    if score is None and result is not None:
        score = result
    # print(f"epd: {line}, stm: {stm}, result: {result}, score: {score}")
    input_data = torch.tensor((stm_features, nstm_features), dtype=torch.float32).to(device)
    output_data = torch.tensor((result, score), dtype=torch.float32).to(device)
    return input_data, output_data


def parse_result(result, stm):
    if '1-0' in result or "1.0" in result:
        return 1.0 if stm == 1 else 0.0
        # return 1.0
    elif '0-1' in result or "0.0" in result:
        return 0.0 if stm == 1 else 1.0
        # return 0.0
    elif '1/2-1/2' in result or "0.5" in result:
        return 0.5
    else:
        return None


def parse_score(score, stm):
    #score = wdl.cp_to_wdl(score)
    return int(score) if stm == 1 else -int(score)
    # return score
