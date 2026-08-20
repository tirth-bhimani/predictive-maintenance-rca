import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import copy


# Settings

TRAIN_PATH = "data/processed/train_processed.csv"
VAL_PATH = "data/processed/val_processed.csv"

MODEL_PATH = "src/models/lstm_rul_model.pth"

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64

EPOCHS = 40
LEARNING_RATE = 0.001

PATIENCE = 5


# Device

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load data

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)


# Sort by engine and cycle
train_df = train_df.sort_values(
    ["unit", "cycle"]
)

val_df = val_df.sort_values(
    ["unit", "cycle"]
)


# Select features

DROP_COLUMNS = ["unit", "RUL"]

feature_columns = [
    column
    for column in train_df.columns
    if column not in DROP_COLUMNS
]

X_train = train_df[feature_columns].values
y_train = train_df["RUL"].values

X_val = val_df[feature_columns].values
y_val = val_df["RUL"].values


# Scale features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)


# Dataset for sliding windows

class RULDataset(Dataset):

    def __init__(self, df, features, targets):

        self.sequences = []
        self.targets = []

        start = 0

        # Process each engine separately
        for unit in df["unit"].unique():

            unit_rows = df["unit"] == unit

            unit_features = features[unit_rows]
            unit_targets = targets[unit_rows]

            # Create sliding windows
            for i in range(
                SEQUENCE_LENGTH,
                len(unit_features) + 1
            ):

                sequence = unit_features[
                    i - SEQUENCE_LENGTH:i
                ]

                target = unit_targets[i - 1]

                self.sequences.append(sequence)
                self.targets.append(target)

        self.sequences = np.array(
            self.sequences,
            dtype=np.float32
        )

        self.targets = np.array(
            self.targets,
            dtype=np.float32
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence = torch.tensor(
            self.sequences[index]
        )

        target = torch.tensor(
            self.targets[index]
        )

        return sequence, target


# Create datasets

train_dataset = RULDataset(
    train_df,
    X_train,
    y_train
)

val_dataset = RULDataset(
    val_df,
    X_val,
    y_val
)


# Create DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print(
    "Training sequences:",
    len(train_dataset)
)

print(
    "Validation sequences:",
    len(val_dataset)
)


# LSTM Model

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Take output from final cycle
        last_output = output[:, -1, :]

        rul = self.fc(last_output)

        return rul.squeeze(1)


# Create model

model = LSTMModel(
    input_size=len(feature_columns)
)

model = model.to(device)


# Loss and optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training

best_val_loss = float("inf")

best_model = None

patience_counter = 0


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # ----------------------------------------------
    # Early stopping
    # ----------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model = copy.deepcopy(
            model.state_dict()
        )

        patience_counter = 0

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping!")

            break


# Load best model

model.load_state_dict(
    best_model
)


# Save checkpoint

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": len(feature_columns),
        "sequence_length": SEQUENCE_LENGTH
    },
    MODEL_PATH
)

print(
    "Model saved:",
    MODEL_PATH
)


# Calculate validation RMSE

model.eval()

predictions = []
actual = []


with torch.no_grad():

    for X_batch, y_batch in val_loader:

        X_batch = X_batch.to(device)

        output = model(X_batch)

        predictions.extend(
            output.cpu().numpy()
        )

        actual.extend(
            y_batch.numpy()
        )


rmse = mean_squared_error(
    actual,
    predictions
) ** 0.5


print()
print("-------------------------")
print("LSTM Validation RMSE")
print("-------------------------")
print(f"RMSE: {rmse:.2f} cycles")