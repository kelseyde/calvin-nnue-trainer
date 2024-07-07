import time


def merge(input_file_names, output_file_name):
    output_file = open(output_file_name, "w")

    num_positions = 0
    start = time.time()
    for file_name in input_file_names:
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("[", "\"")
                line = line.replace("]", "\"")
                line = line.replace("1-0", "1.0")
                line = line.replace("0-1", "0.0")
                line = line.replace("1/2-1/2", "0.5")
                output_file.write(line)
                if num_positions % 100000 == 0:
                    print(f"Processed {num_positions} positions in {time.time() - start:.2f}s")
                num_positions += 1
    print(f"Finished processing {num_positions} positions in {time.time() - start:.2f}s")
