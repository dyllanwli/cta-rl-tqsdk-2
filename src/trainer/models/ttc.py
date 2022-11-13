# Transformer Time series classification 
import logging 
import os 
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['__MODIN_AUTOIMPORT_PANDAS__'] = '1' # Fix modin warning 

import pandas 
import modin.pandas as pd
import ray
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, models
import keras_tuner as kt
import numpy as np
from datetime import datetime 
from tqdm import tqdm
import pytz
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from wandb.keras import WandbCallback

from utils.constant import mlp_units_dict, low_by_label_length, high_by_label_length

class TTCModel:
    def __init__(self, interval: str, commodity_name: str, max_encode_length: int = 60, max_label_length: int = 5):
        ray.init(num_cpus = 62, include_dashboard=False)
        print('GPU name: ', tf.config.list_physical_devices('GPU'))
        self.project_name = "ts_prediction_2"
        self.commodity_name = commodity_name
        self.interval = interval
        self.max_encode_length = max_encode_length
        self.max_label_length = max_label_length

        self.n_classes = 5
        self.train_col_name = ["open", "high", "low", "close", "vol", "open_oi", "close_oi", "is_daytime"]
        self.fit_config = {
            "batch_size": 256,
            "epochs": 100,
            "validation_split": 0.25,
            "shuffle": True,
        }
        self.X_output_path = "./tmp/X_{}{}_{}_{}.npy".format(self.commodity_name, self.interval, self.max_encode_length, self.max_label_length)
        self.y_output_path = "./tmp/y_{}{}_{}_{}.npy".format(self.commodity_name, self.interval, self.max_encode_length, self.max_label_length)
        
    def set_training_data(self, data: pd.DataFrame, debug_mode: bool = False):
        if isinstance(data, pandas.DataFrame):
            X, y = self.pre_process_data(data)
            c = np.array(np.unique(y, return_counts=True)).T
            print("Class distribution: ", c)
            print("Saving data")
            np.save(self.X_output_path, X)
            np.save(self.y_output_path, y)
        else:
            X = np.load(self.X_output_path)
            y = np.load(self.y_output_path)
        if debug_mode:
            X = X[:10000]
            y = y[:10000]
        X = self.timeseries_normalize(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        del X, y
        self.input_shape = self.X_train.shape[1:]
    
    def set_predict_data(self, data: pd.DataFrame):
        print("Set predict data")
        X_predict, y = self.pre_process_data(data)
        c = np.array(np.unique(y, return_counts=True)).T
        print("Class distribution: ", c)
        # np.save("./tmp/X_predict.npy", X)
        # np.save("./tmp/y_predict.npy", y)
        return X_predict, y

    def timeseries_normalize(self, data: np.ndarray):
        print("Normalizing data", data.shape)
        normalize = lambda subset: minmax_scale(subset, feature_range=(0, 2), axis=0)
        for i in tqdm(range(data.shape[0]), bar_format="{percentage:.0f}%"):
            data[i] = normalize(data[i])
        return data
    
    def _process_datatime(self, df: pd.DataFrame):
        print("Processing datetime")
        exchange_tz = pytz.timezone('Asia/Shanghai')
        df["datetime"] =  df["datetime"].apply(lambda x: datetime.utcfromtimestamp(x.value / 1e9).astimezone(exchange_tz))
        df["is_daytime"] = df["datetime"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        return df
    
    def set_volatility_label(self, df: pd.DataFrame):
        """
            check if the price changes by percentage within max_label_length step
            n_classes = 5
            label range:
                -inf ~ -high | -high ~ -low | -low ~ low | low ~ high | high ~ inf
            e.g.
                -inf ~ -0.01 | -0.01 ~ -0.005 | -0.005 ~ 0.005 | 0.005 ~ 0.01 | 0.01 ~ inf
        """
        # df = pd.DataFrame(df)
        low = low_by_label_length[self.interval][self.max_label_length]
        high = high_by_label_length[self.interval][self.max_label_length]

        # reset the index from 0 to df.shape[0]
        df = df.reset_index(drop=True).reset_index()

        def check_volatility(v):
            if v < -high:
                return 0
            elif -high < v < -low:
                return 1
            elif -low < v < low:
                return 2
            elif low < v < high:
                return 3
            elif high < v:
                return 4
            else:
                return np.nan

        numerator = df['close'].to_numpy()[self.max_label_length:]
        denominator = df['close'].to_numpy()[:-self.max_label_length]
        vol = numerator / denominator - 1
        df = df.iloc[:-self.max_label_length]
        df['vol'] = vol
        df['vol'] = df['vol'].apply(lambda x: check_volatility(x))
        print(df["vol"].value_counts())
        df = df[self.train_col_name].dropna()
        return df
    
    def pre_process_data(self, df: pd.DataFrame):
        # start preprocessing

        def process_by_group(g) -> pd.DataFrame:    
            print("Setting volatility label")    
            g = self.set_volatility_label(g)
            target_list = []
            train_list = []
            for i in tqdm(range(g.shape[0] - self.max_encode_length)):
                # yield training, target 
                train, target = g.iloc[i:i+self.max_encode_length].to_numpy(dtype=np.float32), g.iloc[i+self.max_encode_length]["vol"]
                train_list.append(train)
                target_list.append(target)
            return train_list, target_list

        print("Start preprocessing data")
        df = pd.DataFrame(df)
        df = self._process_datatime(df)
        train = []
        target = []

        print("Processing group")
        for g_name, g in df.groupby("underlying_symbol"):
            print("Processing group: ", g_name)
            x, y = process_by_group(g)
            train += x
            target += y

        print("Preprocess data done.")
        return np.array(train), np.array(target)

    def build_model(
        self,
        input_shape,
        head_size: int,
        num_heads: int,
        ff_dim: int,
        num_transformer_blocks: int,
        mlp_units: list,
        dropout=0,
        mlp_dropout=0,
        lstm_units: int = 0,
        hp = False,
    ) -> Model:
        if hp:
            # hyperparameter tuning
            # head_size = hp.Choice("head_size", values=[128, 256, 512, 1024], default=128)
            num_heads = hp.Int("num_heads", min_value=1, max_value=6, step=1)
            ff_dim = hp.Int("ff_dim", min_value=2, max_value=8, step=1)
            mlp_dropout = hp.Float("mlp_dropout", min_value=0.2, max_value=0.4, step=0.1)
            dropout = hp.Float("dropout", min_value=0.2, max_value=0.4, step=0.05)
            num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=3, max_value=8, step=1)
            mlp_units_key = hp.Choice("mlp_units", values=[0,1,2,3,4,5,6,7], default=0)
            mlp_units = mlp_units_dict[mlp_units_key]
            lstm_units = hp.Choice("lstm_units", values=[0, 128, 256], default=0)
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        
        if lstm_units != 0:
            x = layers.LSTM(128, return_sequences=True)(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        return Model(inputs, outputs)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Create a single transformer encoder block.
        """
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res
    
    def model_builder(self, hp = False) -> Model:
        """
        Build the model with hyperparameters
        """
        model = self.build_model(
            self.input_shape,
            head_size=512,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=6,
            mlp_units=[256, 128],
            mlp_dropout=0.3,
            dropout=0.3,
            lstm_units=0,
            hp = hp,
        )

        lr = 1e-4

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["sparse_categorical_accuracy"],
        )
        model.summary()
        return model
    
    def tune(self, search_data_ratio: float = 0.5):
        """
        Tune hyperparameters
        """
        wandb.init(project=self.project_name, group="tune")
        tuner = kt.Hyperband(self.model_builder,
                     objective='sparse_categorical_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='keras_tuner',
                     project_name='ttc_tuner_{}_{}_{}_{}'.format(self.commodity_name, self.interval, self.max_encode_length, self.max_label_length))

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"), 
            WandbCallback(save_model=False, monitor="sparse_categorical_accuracy", mode="max"),
            # tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        ]
        print("Start tuning")
        tuner.search(
            self.X_train[:int(len(self.X_train)*search_data_ratio)],
            self.y_train[:int(len(self.y_train)*search_data_ratio)], 
            epochs=500, validation_split=self.fit_config["validation_split"],
            callbacks=callbacks, shuffle=True, batch_size=self.fit_config["batch_size"])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_config = best_hps.get_config()['values']
        print("Best config", best_hps_config)
        # set config to wandb
        wandb.config.update(best_hps_config)

        # model = tuner.hypermodel.build(best_hps)
        # history = model.fit(self.X_train, self.y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        hypermodel.fit(
            self.X_train,
            self.y_train,
            validation_split=self.fit_config["validation_split"],
            epochs=self.fit_config["epochs"],
            batch_size=self.fit_config["batch_size"],
            shuffle=self.fit_config["shuffle"], 
            callbacks=callbacks,
        )

        eval_result = hypermodel.evaluate(self.X_test, self.y_test)
        print("[test loss, test accuracy]:", eval_result)

    def train(self):
        wandb.init(project=self.project_name, group="train")
        model = self.model_builder()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
            WandbCallback(save_model=True, monitor="sparse_categorical_accuracy", mode="max")
        ]

        model.fit(
            self.X_train,
            self.y_train,
            validation_split=self.fit_config["validation_split"],
            epochs=self.fit_config["epochs"],
            batch_size=self.fit_config["batch_size"],
            shuffle=self.fit_config["shuffle"], 
            use_multiprocessing=self.fit_config["use_multiprocessing"],
            callbacks=callbacks,
        )

        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)
        return model

    def predict(self, model, X_predict, y):
        if isinstance(model, str):
            model: Model = models.load_model(model)
        # wandb.init(project=self.project_name, group="predict")

        X_predict_norm = self.timeseries_normalize(X_predict)

        eval_result = model.evaluate(X_predict_norm, y)
        print("[predict loss, predict accuracy]:", eval_result)

        print("X_predict_norm shape: ", X_predict_norm.shape)
        for x in X_predict_norm:
            predict = model(np.array([x]), training=False)
            # close_price = X_predict[i, -1, 3]
            # print(predict, close_price, y)
            break
        print(predict)

