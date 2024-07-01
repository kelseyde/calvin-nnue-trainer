import time

from torch.utils.data import DataLoader

from dataset import LabelledDataset
from fen import fen_to_features


def load(file_path, batch_size=64, max_size=None):
    """
    Load data from EPD file.

    :param file_path: str, path to EPD file.
    :param max_size: maximum number of positions to load
    :return: list, list of data.
    """
    all_data = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        if max_size is None:
            max_size = len(data)
        for x in range(max_size):
            line = data[x]
            parts = line.split('\"')
            fen_string = parts[0].strip()
            input_data = fen_to_features(fen_string)
            output_data = parse_result(parts[1])
            all_data.append((input_data, output_data))

    num_data = len(all_data)
    training_data = LabelledDataset(all_data[:int(0.9 * num_data)])
    validation_data = LabelledDataset(all_data[int(0.9 * num_data):])
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader


def parse_result(result):
    """
    Parse EPD result.

    :param result: str, EPD result.
    :return: tuple, (result, eval).
    """
    if '1-0' in result:
        return 1.0
    elif '0-1' in result:
        return 0.0
    elif '1/2-1/2' in result:
        return 0.5
    else:
        raise ValueError('Invalid result: {}'.format(result))

