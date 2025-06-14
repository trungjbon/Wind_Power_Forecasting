import numpy as np
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
import matplotlib.pyplot as plt
from ipywidgets import interact
from typing import List, Callable, Tuple
from sklearn import metrics
import tensorflow as tf
from dataclasses import dataclass

FONT_SIZE_TICKS = 15
FONT_SIZE_TITLE = 25
FONT_SIZE_AXES = 20


def top_n_turbines(raw_data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Keeps only the top n turbines that produced more energy on average.
    """
    sorted_patv_by_turbine = (
        raw_data.groupby("TurbID")["Patv"].mean().sort_values(ascending=False))

    top_turbines = list(sorted_patv_by_turbine.index)[:n]
    filtered_data = raw_data.loc[raw_data["TurbID"].isin(top_turbines)]

    print(f"Original data has {len(raw_data)} rows from {len(raw_data.TurbID.unique())} turbines.\n")
    print(f"Sliced data has {len(filtered_data)} rows from {len(filtered_data.TurbID.unique())} turbines.")

    return filtered_data


def format_datetime(df: pd.DataFrame, initial_date_str: str) -> pd.DataFrame:
    """
    Formats Day and Tmstamp features into a Datetime feature.
    """
    data = df.copy()

    initial_date = pd.to_datetime(initial_date_str, format="%d %m %Y")
    data["Date"] = initial_date + pd.to_timedelta(data["Day"] - 1, unit="D")
    data["Datetime"] = pd.to_datetime(data["Date"].astype(str) + " " + data["Tmstamp"])
    
    data.drop(columns=["Day", "Tmstamp", "Date"], inplace=True)
    cols = ["Datetime"] + [col for col in data.columns if col != "Datetime"]
    data = data[cols]

    return data


def histogram_plot(df: pd.DataFrame, features: List[str], bins: int=16):
    """
    Create interactive histogram plots.
    """
    turbine_ids = df["TurbID"].unique()
    turbine_data = {tid: df.loc[df["TurbID"] == tid] for tid in turbine_ids}

    def _plot(turbine, feature):
        data = turbine_data[turbine]
        x = data[feature]

        plt.figure(figsize=(8, 5))
        plt.hist(x, bins=bins, edgecolor="black")
        plt.xlabel(f"{feature}", fontsize=FONT_SIZE_AXES)
        plt.ylabel(f"Count", fontsize=FONT_SIZE_AXES)
        plt.title(f"Feature: {feature} - Turbine: {turbine}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(options=turbine_ids, description="Turbine")
    feature_selection = widgets.Dropdown(options=features, description="Feature")

    interact(_plot, turbine=turbine_selection, feature=feature_selection)



def scatter_plot(df: pd.DataFrame, features: List[str]):
    """
    Creates interactive scatter plots of the data.
    """
    turbine_ids = df["TurbID"].unique()

    def _plot(turbine, var_x, var_y):
        data = df.loc[df["TurbID"] == turbine]
        x = data[var_x]
        y = data[var_y]

        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, s=9, c='blue', alpha=0.5)
        plt.xlabel(var_x, fontsize=FONT_SIZE_AXES)
        plt.ylabel(var_y, fontsize=FONT_SIZE_AXES)
        plt.title(f"Scatter plot of {var_x} against {var_y}", fontsize=FONT_SIZE_TITLE)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        plt.show()

    turbine_selection = widgets.Dropdown(options=turbine_ids, description="Turbine")
    x_var_selection = widgets.Dropdown(options=features, description="X-Axis")
    y_var_selection = widgets.Dropdown(options=features, description="Y-Axis", value="Patv")

    interact(_plot, turbine=turbine_selection, var_x=x_var_selection, var_y=y_var_selection)

# Plots correlation matrix for a given dataset.
def correlation_matrix(data: pd.core.frame.DataFrame):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, cbar=False, cmap="RdBu", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Features")
    plt.show()


def tag_abnormal_values(df: pd.DataFrame, conditions: List[pd.Series]) -> pd.DataFrame:
    """
    Determines if a given record is an abnormal value.
    """
    data = df.copy()

    data["Include"] = True

    for condition in conditions:
        data.loc[condition, "Include"] = False
        
    return data



def fix_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces very low temperature values with linear interpolation.
    """
    data = df.copy()
    for feature in ["Etmp", "Itmp"]:
        q = data[feature].quantile(0.01)
        data.loc[data[feature] < q, feature] = np.nan
        data[feature] = data[feature].interpolate()

    return data


def cut_pab_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deletes redundant Pab features from dataset.
    """
    data = df.copy()
    data.drop(columns=["Pab2", "Pab3"], inplace=True)
    data.rename(columns={"Pab1": "Pab"}, inplace=True)

    return data


def transform_angles(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Transform angles into their Sin/Cos encoding.
    """
    data = df.copy()
    # np.cos and np.sin expect angles in radians
    rads = np.radians(data[feature])

    # Compute Cos and Sin
    data[f"{feature}Cos"] = np.cos(rads)
    data[f"{feature}Sin"] = np.sin(rads)

    data.drop(columns=feature, inplace=True)

    return data


def fix_active_powers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix negative active powers
    """
    data = df.copy()
    # data.loc[data["Patv"] < 0, "Patv"] = 0
    data["Patv"] = data["Patv"].apply(lambda x: max(0, x))

    return data


def prepare_data(df: pd.DataFrame, turb_id: int) -> pd.DataFrame:
    """
    Pre-process data before feeding to neural networks for training.
    - Resampling to an hourly basis
    - Using data from a single turbine
    - Format datetime
    - Mask abnormal values
    - Re-order columns
    """
    data = df.copy()

    data = data[5::6]
    data = data.loc[data["TurbID"] == turb_id]
    data.drop(columns=["TurbID"], inplace=True)
    data.index = pd.to_datetime(data.pop("Datetime"), format="%Y-%m-%d %H:%M")
    data = data.mask(data["Include"] == False, 0)
    data.drop(columns=["Include"], inplace=True)

    cols = [
        "Wspd", "Etmp", "Itmp", "Prtv", "WdirCos", "WdirSin", 
        "NdirCos", "NdirSin", "PabCos", "PabSin", "Patv"
    ]

    data = data[cols]

    return data


def normalize_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Normalizes train, val and test splits.
    """
    train_mean = train_data.mean()
    train_std = train_data.std()

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, val_data, test_data, train_mean, train_std


@dataclass
class DataSplits:
    """Class to encapsulate normalized/unnormalized train, val, test, splits."""
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    test_data: pd.DataFrame
    train_mean: pd.Series
    train_std: pd.Series
    train_df_unnormalized: pd.DataFrame
    val_df_unnormalized: pd.DataFrame
    test_df_unnormalized: pd.DataFrame


def train_val_test_split(df: pd.DataFrame) -> DataSplits:
    """
    Splits a dataframe into train, val and test.
    """
    n = len(df)
    train_df = df.iloc[0 : int(n * 0.7)]
    val_df = df.iloc[int(n * 0.7) : int(n * 0.9)]
    test_df = df.iloc[int(n * 0.9) :]

    train_df_un = train_df.copy()
    val_df_un = val_df.copy()
    test_df_un = test_df.copy()

    train_df, val_df, test_df, train_mn, train_st = normalize_data(train_df, val_df, test_df)

    ds = DataSplits(
        train_data=train_df,
        val_data=val_df,
        test_data=test_df,
        train_mean=train_mn,
        train_std=train_st,
        train_df_unnormalized=train_df_un,
        val_df_unnormalized=val_df_un,
        test_df_unnormalized=test_df_un,
    )

    return ds

def compute_metrics(true_series: np.ndarray, forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes MSE, MAE, ,RMSE and R2 between two time series.
    """
    mse = metrics.mean_squared_error(true_series, forecast)
    mae = metrics.mean_absolute_error(true_series, forecast)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(true_series, forecast)

    return mse, mae, rmse, r2

class WindowGenerator:
    """
    Class that handles all of the windowing and plotting logic for time series.
    """
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = ["Patv"]

        self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def plot_long(self, model, data_splits, plot_col="Patv", time_steps_future=1, baseline_mae=None):
        train_mean, train_std = data_splits.train_mean, data_splits.train_std
        test_size = len(self.test_df)
        test_data = self.make_dataset(data=self.test_df, batch_size=test_size, shuffle=False)

        inputs, labels = next(iter(test_data))

        plt.figure(figsize=(20, 6))
        label_col_index = self.label_columns_indices.get(plot_col, None)

        labels = self.denormalize(labels, train_std[plot_col], train_mean[plot_col])

        upper = self.label_width - (time_steps_future - 1)
        lower = self.label_indices[-1] - upper
        label_indices_long = self.test_df.index[lower : -upper]

        plt.plot(
            label_indices_long,
            labels[:, time_steps_future - 1, label_col_index],
            label="Labels",
            c="green")

        if (model is not None):
            predictions = model(inputs)
            predictions = self.denormalize(predictions, train_std[plot_col], train_mean[plot_col])
            predictions_for_timestep = predictions[:, time_steps_future - 1, label_col_index]
            predictions_for_timestep = tf.nn.relu(predictions_for_timestep).numpy()

            plt.plot(label_indices_long, predictions_for_timestep, label="Predictions", c="orange", linewidth=3)
            plt.legend(fontsize=FONT_SIZE_TICKS)

            mse, mae, rmse, r2 = compute_metrics(
                labels[:, time_steps_future - 1, label_col_index].numpy(), predictions_for_timestep)

            if (baseline_mae is None):
                baseline_mae = mae

            print(f"\nMean Absolute Error (kW): {mae:.2f} for forecast.")
            print(f"\nRoot Mean Squared Error (kW): {rmse:.2f} for forecast.")
            print(f"\nCoefficient of Determination (kW): {r2:.2f} for forecast.")

        plt.title("Predictions vs Real Values for Test Split", fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Date", fontsize=FONT_SIZE_AXES)
        plt.ylabel(f"{plot_col} (kW)", fontsize=FONT_SIZE_AXES)
        plt.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
        return mae

    def make_dataset(self, data, batch_size=32, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size)

        return ds.map(self.split_window)
    
    def denormalize(self, series, std, mean):
        return (series * std) + mean

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


def generate_window(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, days_in_past: int, width: int = 24
        ) -> WindowGenerator:
    """
    Creates a windowed dataset given the train, val, test splits and the number of days into the past.
    """
    OUT_STEPS = 24
    multi_window = WindowGenerator(
        input_width=width * days_in_past,
        label_width=OUT_STEPS,
        shift=OUT_STEPS,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )
    return multi_window


def create_model(num_features: int, days_in_past: int) -> tf.keras.Model:
    """
    Creates a Conv-LSTM model for time series prediction.
    """
    CONV_WIDTH = 3
    OUT_STEPS = 24

    model = tf.keras.Sequential(
        [
            # Input shape
            tf.keras.layers.Input(shape=(days_in_past * 24, num_features)),

            # CNN layers for feature extraction
            tf.keras.layers.Conv1D(filters=256, kernel_size=CONV_WIDTH, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=256, kernel_size=CONV_WIDTH, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),

            # LSTM layers for sequential learning
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False)),

            # Fully connected layers
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            # Output layer
            tf.keras.layers.Dense(OUT_STEPS, activation="linear"),
            tf.keras.layers.Reshape([OUT_STEPS, 1]),
        ]
    )

    return model

def compile_and_fit(model: tf.keras.Model, window: WindowGenerator, patience: int = 10) -> tf.keras.callbacks.History:
    """
    Compiles and trains a model given a patience threshold.
    """
    EPOCHS = 20

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", restore_best_weights=True)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam())

    history = model.fit(
        window.train, epochs=EPOCHS, validation_data=window.val, callbacks=early_stopping)

    if (len(history.epoch) < EPOCHS):
        print("\nTraining stopped early to prevent overfitting.")

    return history


def train_conv_lstm_model(data: pd.DataFrame, features: List[str], days_in_past: int) -> Tuple[WindowGenerator, tf.keras.Model, DataSplits]:
    """
    Trains the Conv-LSTM model for time series prediction.
    """
    data_splits = train_val_test_split(data[features])

    train_data, val_data, test_data = (
        data_splits.train_data,
        data_splits.val_data,
        data_splits.test_data
    )

    window = generate_window(train_data, val_data, test_data, days_in_past)
    num_features = window.train_df.shape[1]

    model = create_model(num_features, days_in_past)
    history = compile_and_fit(model, window)

    return window, model, data_splits, history


def prediction_plot(func: Callable, model: tf.keras.Model, data_splits: DataSplits) -> None:
    """
    Plot an interactive visualization of predictions vs true values.
    """
    def _plot(time_steps_future):
        mae = func(
            model,
            data_splits,
            time_steps_future=time_steps_future,
        )

    time_steps_future_selection = widgets.IntSlider(
        value=24,
        min=1,
        max=24,
        step=1,
        description="Hours into future",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        layout={"width": "500px"},
        style={"description_width": "initial"},
    )

    interact(_plot, time_steps_future=time_steps_future_selection)


def add_wind_speed_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates syntethic wind speed forecasts. The more into the future, the more noise these have.
    """
    df_2 = df.copy()
    np.random.seed(8752)

    for period in range(1, 25):
        noise_level = 2 + 0.05 * period
        noise = abs(np.random.randn(len(df))) * noise_level
        
        padding = df_2["Wspd"][-period:].to_numpy()
        values = np.concatenate((df_2["Wspd"][period:].values, padding)) + noise

        df_2[f"fc-{period}h"] = values

    return df_2