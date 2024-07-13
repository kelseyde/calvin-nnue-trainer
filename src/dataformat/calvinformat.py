from torch.utils.data import Dataset

from src.dataformat.board import PackedBoard


class CalvinDataset(Dataset):

    def __init__(self, file_path, max_size=None):
        self.file_path = file_path
        self.file = open(file_path, 'rb')
        self.max_size = max_size
        self.length = self._compute_length()

    def _compute_length(self):
        # Move to the end of the file to determine its size
        self.file.seek(0, 2)
        file_size = self.file.tell()
        # Each PackedBoard is 43 bytes
        length = file_size // 43
        if self.max_size is not None:
            return min(length, self.max_size)
        return length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        offset = idx * 43
        self.file.seek(offset)
        data = self.file.read(43)
        pb = PackedBoard.from_bytes(data)
        return pb.to_features()

    def close(self):
        self.file.close()