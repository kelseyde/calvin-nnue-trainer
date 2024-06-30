import time

from fen_utils import fen_to_features


def load_from_epd_file(file_path):
    """
    Load data from EPD file.

    :param file_path: str, path to EPD file.
    :return: list, list of data.
    """
    all_data = []
    with open(file_path, 'r') as f:
        data = f.readlines()
        for line in data:
            parts = line.split('\"')
            fen = parts[0].strip()
            input_data = fen_to_features(fen)
            output_data = parse_epd_result(parts[1])
            all_data.append((input_data, output_data))

    num_data = len(all_data)
    training_data = all_data[:int(0.8 * num_data)]
    validation_data = all_data[int(0.8 * num_data):int(0.9 * num_data)]
    test_data = all_data[int(0.9 * num_data):]
    return training_data, validation_data, test_data


def parse_epd_result(result):
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


start = time.time()
file_path = "/Users/kelseyde/git/dan/calvin/calvin-chess-engine/src/test/resources/texel/quiet_positions.epd"
data = load_from_epd_file(file_path)
end= time.time()
print("Complete in ", end-start)
print(len(data))
print(len(data[0]))
print(len(data[1]))
print(len(data[2]))
for i in range(10):
    print(data[0][i])
