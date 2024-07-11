import json

from tqdm import tqdm

INPUT_FILE = "/Users/kelseyde/Downloads/lichess_db_eval.jsonl"
OUTPUT_FILE = "/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/datasets/lichess_db_eval.epd"

count = 0
with open(INPUT_FILE, "r") as f:
    with open(OUTPUT_FILE, "w") as out:
        for line in tqdm(f):
            pos = json.loads(line)
            fen = pos["fen"]
            pv = pos["evals"][0]["pvs"][0]
            if "cp" not in pv:
                continue
            cp = int(pv["cp"])
            out.write(f"{fen},,{cp}\n")
            count += 1
