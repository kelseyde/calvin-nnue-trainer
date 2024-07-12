import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src import model
from src.dataformat.epd import load

INPUT_FILE_PATH = "../datasets/training_data_1.txt"
# PREVIOUS_MODEL = "/Users/kelseyde/git/dan/calvin/calvin-chess-engine/src/main/resources/nnue/256HL-3B5083B8.nnue"
PREVIOUS_MODEL = None
# PREVIOUS_MODEL = None
OUTPUT_FILE_PATH = "/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/yukon_ho_5.nnue"
DEVICE = torch.device("mps")
NUM_WORKERS = 3
NUM_EPOCHS = 100
CHECKPOINT_FREQUENCY = 1
MAX_DATA = None
BATCH_SIZE = 1024
INPUT_SIZE = 768
HIDDEN_SIZE = 16
LEARNING_RATE = 0.01
MOMENTUM = 0.0
STEP_SIZE = 5
GAMMA = 0.1
LAMBDA = 0.75
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
    train_loader, val_loader = load(INPUT_FILE_PATH, batch_size=BATCH_SIZE, device=DEVICE, max_size=MAX_DATA,
                                    delimiter='|', fen_index=0, score_index=1, result_index=2)
    nnue = init_model()
    optimizer = torch.optim.SGD(nnue.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

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
            optimizer.zero_grad()
            epoch_loss += float(error.item())
            loop.set_description(f"epoch: {epoch}, train loss: {error.item():.6f}")

        # scheduler.step()  # Adjust learning rate
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
            print(f"epoch: {epoch}, saving model to {OUTPUT_FILE_PATH}")
            torch.save(nnue.state_dict(), f"../nets/state_dict_{epoch}.pt")
            nnue.save(OUTPUT_FILE_PATH)

    nnue.save(OUTPUT_FILE_PATH)


def init_model():
    if PREVIOUS_MODEL is not None:
        return model.NNUE.load(PREVIOUS_MODEL, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(DEVICE)
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
