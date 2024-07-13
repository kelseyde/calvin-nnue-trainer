import torch

from src.dataformat import epd, board

INPUT_FILE_PATH = "../../datasets/training_data_0.txt"
OUTPUT_FILE_PATH = "../../datasets/calvin_data_1.bin"
LIMIT = None

count = 0
with open(OUTPUT_FILE_PATH, "wb") as out:
    with open(INPUT_FILE_PATH, "r") as in_:
        for line in in_:
            count += 1
            if LIMIT is not None and count > LIMIT:
                break
            if count % 10000 == 0:
                print(f"Processed {count} lines")

            # Parse FEN features and convert to PackedBoard features
            pb = board.PackedBoard.from_labelled(line)

            bts = pb.to_bytes()
            assert len(bts) == 43, "PackedBoard is not 43 bytes long!"
            pb2 = board.PackedBoard.from_bytes(bts)

            pb_features = pb.to_features()
            pb2_features = pb2.to_features()

            # Ensure the features match
            assert torch.equal(pb2_features[0], pb_features[0]), "Feature tensors do not match (3)!"
            assert torch.equal(pb2_features[1], pb_features[1]), "Feature tensors do not match (4)!"

            # Write the packed board features to the output file
            out.write(bts)