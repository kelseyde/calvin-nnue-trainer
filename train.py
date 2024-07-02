import random
import time as t
import timeit

import matplotlib.pyplot as plt
import model
import torch

from epd import load

INPUT_FILE_PATH = "/Users/kelseyde/git/dan/calvin/calvin-chess-engine/src/test/resources/texel/quiet_positions.epd"
OUTPUT_FILE_PATH = "/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/first_attempt.nnue"
DEVICE = torch.device("cpu")
NUM_WORKERS = 1
NUM_EPOCHS = 100
MAX_SIZE = 100000
BATCH_SIZE = 2048
INPUT_SIZE = 768
HIDDEN_SIZE = 256
LEARNING_RATE = 0.001
MOMENTUM = 0.0
STEP_SIZE = 100
GAMMA = 0.1


def visualise(train_losses, validation_losses):
    """Visualise the training and validation loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss over Epochs')
    plt.show()


def main():
    train_loader, val_loader = load(INPUT_FILE_PATH, batch_size=BATCH_SIZE, max_size=MAX_SIZE)
    nnue = model.NNUE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.SGD(nnue.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    loss_fn = torch.nn.MSELoss()

    train_losses = []
    validation_losses = []

    for epoch in range(NUM_EPOCHS):
        start = t.time()

        # Train the model
        nnue.train()
        epoch_loss = 0.0
        for input_data, output_data in train_loader:
            input_data, output_data = input_data.to(DEVICE), output_data.to(DEVICE)
            optimizer.zero_grad()
            predictions = nnue(input_data)
            output_data = output_data.unsqueeze(1)  # Ensure target has the same shape as predictions
            error = loss_fn(predictions, output_data)
            error.backward()
            optimizer.step()
            epoch_loss += error.item()

        scheduler.step()  # Adjust learning rate
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate the model
        nnue.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for input_data, output_data in val_loader:
                input_data, output_data = input_data.to(DEVICE), output_data.to(DEVICE)
                predictions = nnue(input_data)
                output_data = output_data.unsqueeze(1)
                error = loss_fn(predictions, output_data)
                validation_loss += error.item()
        avg_val_loss = validation_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        end = t.time()
        time = "{0:.2f}".format(end - start)
        print(f"epoch: {epoch}, time: {time}s, train error: {avg_train_loss:.6f}, val error: {avg_val_loss:.6f}")

    nnue.save(OUTPUT_FILE_PATH)
    visualise(train_losses, validation_losses)


if __name__ == "__main__":
    main()
