import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/train_FD001.txt"
PROCESSED_PATH = "data/processed"

WINDOW_SIZE = 10

columns = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3"
]

columns += [f"sensor{i}" for i in range(1, 22)]

#load dataset
def load_data():

    df = pd.read_csv(
        RAW_PATH,
        sep=r"\s+",
        header=None,
        names=columns
    )

    print("Raw data shape:", df.shape)

    return df

#compute remaining useful life (RUL) for each row
def calculate_train_rul(df):
    
    #count max cycle for each unit  
    max_cycle = df.groupby("unit")["cycle"].max()

    df = df.merge(
        max_cycle.rename("max_cycle"),
        on="unit"
    )

    df["RUL"] = df["max_cycle"] - df["cycle"]

    df["RUL"] = df["RUL"].clip(upper=125)

    df.drop(columns=["max_cycle"], inplace=True)

    return df



#create features for each unit based on the last 10 cycles
def create_features(df):

    # Get all sensor column names
    sensor_columns = [
        f"sensor{i}"
        for i in range(1, 22)
    ]

    # Create features for every sensor
    for sensor in sensor_columns:

        # Rolling mean
        df[f"{sensor}_mean_10"] = (
            df.groupby("unit")[sensor].transform(
                lambda x: x.rolling(
                    WINDOW_SIZE,
                    min_periods=1
                ).mean()
            )
        )

        # Rolling standard deviation
        df[f"{sensor}_std_10"] = (
            df.groupby("unit")[sensor].transform(
                lambda x: x.rolling(
                    WINDOW_SIZE,
                    min_periods=1
                ).std()
            )
        )

    # First row of every engine can have NaN std
    # Replace NaN values with 0
    df = df.fillna(0)
    return df

def split_by_unit(df):
    
     # Get all unique engine IDs
    units = df["unit"].unique()

    # Make the split reproducible
    np.random.seed(42)

    # Shuffle engine IDs
    np.random.shuffle(units)

    # Use 80% engines for training
    split_point = int(len(units) * 0.8)

    train_units = units[:split_point]

    val_units = units[split_point:]

    # Select complete engines
    train_df = df[
        df["unit"].isin(train_units)
    ].copy()

    val_df = df[
        df["unit"].isin(val_units)
    ].copy()

    return train_df, val_df
# Main preprocessing pipeline
def main():
    os.makedirs(
        PROCESSED_PATH,
        exist_ok=True
    )

    df = load_data()

    df = calculate_train_rul(df)

    df = create_features(df)

    train_df, val_df = split_by_unit(df)

    # Save training data
    train_df.to_csv(
        f"{PROCESSED_PATH}/train_processed.csv",
        index=False
    )

    # Save validation data
    val_df.to_csv(
        f"{PROCESSED_PATH}/val_processed.csv",
        index=False
    )

    print("\nProcessing completed!")

    print(
        "Train shape:",
        train_df.shape
    )

    print(
        "Validation shape:",
        val_df.shape
    )

    print(
        "Train engines:",
        train_df["unit"].nunique()
    )

    print(
        "Validation engines:",
        val_df["unit"].nunique()
    )

    # Check if any engine exists in both datasets
    common_units = set(
        train_df["unit"]
    ).intersection(
        set(val_df["unit"])
    )

    print(
        "Common engines:",
        common_units
    )


# --------------------------------------------------
# 8. Run the program
# --------------------------------------------------

if __name__ == "__main__":
    main()