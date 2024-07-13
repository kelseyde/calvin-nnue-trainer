import matplotlib.pyplot as plt
import torch
import dataloader
from tqdm import tqdm

from src import model

DATASET_DIR = "../datasets/"
DATASET_FILE = "calvin_data_0.bin"
# DATASET_FILE = "training_data_0.txt"
DATASET_PATH = DATASET_DIR + DATASET_FILE
DATA_FORMAT = dataloader.DataFormat.CALVIN
MODEL_DIR = "../nets/"
PREVIOUS_MODEL_NAME = "calvinball_1_3"
CHECKPOINT_MODEL_NAME = "calvinball_2"
CHECKPOINT_MODEL_PATH = MODEL_DIR + CHECKPOINT_MODEL_NAME

DEVICE = torch.device("mps")
NUM_WORKERS = 3
NUM_EPOCHS = 30
CHECKPOINT_FREQUENCY = 1
MAX_DATA = None
BATCH_SIZE = 1024
INPUT_SIZE = 768
HIDDEN_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.0
STEP_SIZE = 10
GAMMA = 0.1
LAMBDA = 0.9
SCALE = 400.0


def train():
    print("starting training session")
    print(f"device: {DEVICE}, "
          f"epochs: {NUM_EPOCHS}, "
          f"batch: {BATCH_SIZE}, "
          f"architecture: ({INPUT_SIZE}->{HIDDEN_SIZE})x2->1, "
          f"learning rate: {LEARNING_RATE}, "
          f"momentum: {MOMENTUM}, "
          f"lambda: {LAMBDA}, "
          f"step size: {STEP_SIZE}, "
          f"gamma: {GAMMA}")

    print("loading training data...")
    train_loader, val_loader = dataloader.load(DATASET_PATH, DATA_FORMAT, BATCH_SIZE, MAX_DATA, DEVICE)
    nnue = model.NNUE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.SGD(nnue.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    train_losses = []
    validation_losses = []

    for epoch in range(NUM_EPOCHS):
        # Train the model
        nnue.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader)
        for input_data, output_data in loop:
            input_data = input_data.to(DEVICE)
            output_data = output_data.to(DEVICE)
            predictions = nnue(input_data)
            error = nnue.loss(predictions, output_data, SCALE, LAMBDA)
            error.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(error.item())
            loop.set_description(f"epoch: {epoch}, train loss: {error.item():.6f}")

        scheduler.step()  # Adjust learning rate
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate the model
        nnue.eval()
        validation_loss = 0.0
        with torch.no_grad():
            loop = tqdm(val_loader)
            for input_data, output_data in loop:
                input_data = input_data.to(DEVICE)
                output_data = output_data.to(DEVICE)
                predictions = nnue(input_data)
                error = nnue.loss(predictions, output_data, SCALE, LAMBDA)
                validation_loss += float(error.item())
                loop.set_description(f"epoch: {epoch},   val loss: {error.item():.6f}")
        avg_val_loss = validation_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f"epoch: {epoch}, train loss: {avg_train_loss:.6f}, val loss: {avg_val_loss:.6f}")

        if epoch % CHECKPOINT_FREQUENCY == 0:
            visualise(train_losses, validation_losses)
            save_name = f"{CHECKPOINT_MODEL_PATH}_{epoch}.pt"
            print(f"epoch: {epoch}, saving model to {save_name}")
            torch.save(nnue.state_dict(), save_name)
            # nnue.save(OUTPUT_FILE_PATH)

    # nnue.save(OUTPUT_FILE_PATH)


def init_model():
    if PREVIOUS_MODEL_NAME is not None:
        nnue = model.NNUE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
        nnue.load_state_dict(torch.load(MODEL_DIR + PREVIOUS_MODEL_NAME + ".pt", map_location=DEVICE))
        return nnue
    else:
        return model.NNUE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)


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


if __name__ == "__main__":
    train()
