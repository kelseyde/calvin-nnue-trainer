import enum
import time as t
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.dataformat.calvinformat import CalvinDataset
from src.dataformat.epd import EPDFileDataset


class DataFormat(enum.Enum):
    EPD = 0
    CALVIN = 1


def load(file_path, data_format=DataFormat.CALVIN, batch_size=64, max_size=None, device="mps"):
    start = t.time()

    dataset = None
    match data_format:
        case DataFormat.EPD:
            dataset = EPDFileDataset(file_path, max_size, delimiter='|', fen_index=0, score_index=1, result_index=2)
        case DataFormat.CALVIN:
            dataset = CalvinDataset(file_path, max_size=max_size)

    num_data = len(dataset)
    train_size = int(0.9 * num_data)
    val_size = num_data - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    end = t.time()
    print(f"data loaded in {end - start:.2f}s")
    print(f"data size: {num_data}")

    return train_loader, val_loader
