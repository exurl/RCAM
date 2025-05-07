import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

# -----------------------------
# Step 1: LSTM Model Definition
# -----------------------------

class InverseDynamicsLSTM(nn.Module):
    def __init__(self, state_dim, hidden_dim, control_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, control_dim)

    def forward(self, x_padded, lengths):
        # Sort by decreasing length
        lengths, sort_idx = lengths.sort(descending=True)
        x_padded = x_padded[sort_idx]

        # Pack and run through LSTM
        packed = pack_padded_sequence(x_padded, lengths, batch_first=True)
        packed_out, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Apply linear layer to each timestep
        u_hat = self.fc(unpacked)

        # Unsort to match original batch order
        _, unsort_idx = sort_idx.sort()
        u_hat = u_hat[unsort_idx]

        return u_hat

if __name__ == "__main__":  # Ensures that training is not performed when importing model class from another file

    # -----------------------------
    # Step 2: Import RCAM data
    # -----------------------------

    RCAM_data = np.load("RCAM_data.npy", allow_pickle=True).item()

    # -----------------------------
    # Step 3: Dataset and Dataloader
    # -----------------------------

    class RCAMDataset(Dataset):
        def __init__(self, data_dict):
            self.samples = []
            for trim_case in data_dict.values():
                for profile in trim_case.values():
                    x_raw = profile["x"]
                    u_raw = profile["u"]
                    x = preprocess(x_raw, augment_dxdt=True)  # Preprocess state variables
                    # Remove psi terms from state vector to ensure model dynamics are invariant to heading angle
                    x = np.delete(x, [8, 12], axis=1) 
                    u = preprocess(u_raw)  # Preprocess control variables
                    self.samples.append((x, u, len(x)))

        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            x, u, length = self.samples[idx]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), length

    def collate_fn(batch):
        x_list, u_list, lengths = zip(*batch)
        lengths = torch.tensor(lengths, dtype=torch.long)
        x_padded = pad_sequence(x_list, batch_first=True)  # input: state
        u_padded = pad_sequence(u_list, batch_first=True)  # output: control
        return x_padded, u_padded, lengths

    dataset = RCAMDataset(RCAM_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # -----------------------------
    # Step 4: Training Loop
    # -----------------------------

    # Define the model
    state_dim = 27  # Adjusted state dimension after augmentation
    control_dim = 5
    hidden_dim = 64
    model = InverseDynamicsLSTM(state_dim, hidden_dim, control_dim)

    criterion = nn.MSELoss(reduction='none')  # so we can mask padded outputs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Record loss and accuracy for every epoch.
    all_train_loss = []
    all_train_accuracy = []

    num_epochs = 100  # Set number of training epochs

    for epoch in tqdm(range(num_epochs)):
        
        # Prepare the model for training.
        model.train()

        # Initialize this epoch's loss and accuracy results.
        train_loss = 0.0
        
        for x_batch, u_batch, lengths in dataloader:
            optimizer.zero_grad()   # Zero the gradient buffers

            u_pred = model(x_batch, lengths)  # Predict control inputs from state trajectory

            # Mask padded values
            max_len = u_batch.size(1)
            mask = torch.arange(max_len)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1).expand_as(u_batch)

            loss = criterion(u_pred, u_batch)
            loss = loss[mask].mean()

            loss.backward()     # Compute gradients
            optimizer.step()    # Update the model weights

            train_loss += loss.item()   # Record the loss

        # Record the average loss across batches for this epoch
        all_train_loss.append(train_loss / len(dataloader))

    # Plot the loss across epochs
    plt.figure(figsize=(8, 3))
    plt.plot(all_train_loss, "o-", color="tab:red", label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save the model parameters
    output_file_path = "model_parameters_lstm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': all_train_loss,
        'epoch': epoch
    }, output_file_path)
    print(f"Saved model parameters to {output_file_path}")
    print(f"Final training loss: {all_train_loss[-1]}")
