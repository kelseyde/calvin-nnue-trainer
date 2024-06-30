import time

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data_loader import load_from_epd_file
from dataset import LabelledDataset
from nnue import NNUE

DEVICE = torch.device("cpu")
NUM_WORKERS = 1
MAX_SIZE = None
BATCH_SIZE = 64
INPUT_SIZE = 768
HIDDEN_SIZE = 256
LEARNING_RATE = 0.001
MOMENTUM = 0.0
STEP_SIZE = 500
GAMMA = 0.1


# Check if MPS is available
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = DEVICE

# Ensure PyTorch uses all available CPU resources
torch.set_num_threads(torch.get_num_threads())  # Use all available threads
torch.set_num_interop_threads(torch.get_num_threads())

file_path = "/Users/kelseyde/git/dan/calvin/calvin-chess-engine/src/test/resources/texel/quiet_positions.epd"
training_data, validation_data = load_from_epd_file(file_path, max_size=MAX_SIZE)

train_dataset = LabelledDataset(training_data)
validation_dataset = LabelledDataset(validation_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False,  pin_memory=True)

model = NNUE(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)  # Move model to MPS or CPU

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
loss_fn = torch.nn.MSELoss()

train_losses = []
validation_losses = []

for epoch in range(100):
    start = time.time()

    # Train the model
    model.train()
    epoch_loss = 0.0
    for input_data, output_data in train_loader:
        input_data, output_data = input_data.to(device), output_data.to(device)  # Move data to MPS or CPU
        optimizer.zero_grad()
        predictions = model(input_data)
        output_data = output_data.unsqueeze(1)  # Ensure target has the same shape as predictions
        error = loss_fn(predictions, output_data)
        error.backward()
        optimizer.step()
        epoch_loss += error.item()

    scheduler.step()  # Adjust learning rate
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validate the model
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for input_data, output_data in validation_loader:
            input_data, output_data = input_data.to(device), output_data.to(device)  # Move data to MPS or CPU
            predictions = model(input_data)
            output_data = output_data.unsqueeze(1)
            error = loss_fn(predictions, output_data)
            validation_loss += error.item()
    avg_validation_loss = validation_loss / len(validation_loader)
    validation_losses.append(avg_validation_loss)

    end = time.time()
    duration = ("{0:.2f}".format(end - start))
    lr = optimizer.param_groups[0]['lr']
    print("epoch: {}, time: {}s, lr: {}, train error: {:.6f}, val error: {:.6f}".format(epoch, duration, lr, avg_train_loss, avg_validation_loss))

# Plotting the loss graph
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss over Epochs')
plt.show()