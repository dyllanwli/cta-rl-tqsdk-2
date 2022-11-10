# Transformer Time series classification 
import logging 
import os 
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import pandas as pd
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
from sklearn.preprocessing import MinMaxScaler
from wandb.keras import WandbCallback

from utils.constant import mlp_units_dict

class TTCModel:
    def __init__(self, data: pd.DataFrame):
        print('GPU name: ', tf.config.list_physical_devices('GPU'))
        self.n_classes = 5
        self.train_col_name = ["open", "high", "low", "close", "vol", "open_oi", "close_oi", "is_daytime"]
        max_encoder_length = 90
        max_label_length = 10
        if isinstance(data, pd.DataFrame):
            X, y = self.pre_process_saving(data, max_encoder_length, max_label_length)
            c = np.array(np.unique(y, return_counts=True)).T
            print("Class distribution: ", c)
            print("Saving data")
            np.save("./tmp/X.npy", X)
            np.save("./tmp/y.npy", y)
        else:
            X, y = data
        X = self.timeseries_normalize(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
        self.input_shape = self.X_train.shape[1:]
    
    def timeseries_normalize(self, data: np.ndarray):
        print("Start normalizing data")
        for subset in tqdm(range(data.shape[0])):
            # subset shpae: (max_encode_length, 8)
            scaler = MinMaxScaler(feature_range=(0, 2))
            data[subset] = scaler.fit_transform(data[subset])
            # print(data[subset])
        return data
    
    def pre_process_saving(self, data: pd.DataFrame, max_encode_length: int, max_label_length: int):
        def set_volatility_label(df: pd.DataFrame, max_label_length):
            """
                check if the price changes by percentage within max_label_length step
                n_classes = 5
                label range:
                    -inf ~ -high | -high ~ -low | -low ~ low | low ~ high | high ~ inf
                e.g.
                    -inf ~ -0.01 | -0.01 ~ -0.005 | -0.005 ~ 0.005 | 0.005 ~ 0.01 | 0.01 ~ inf
            """
            low = 0.005/3
            high = 0.01/3

            # reset the index from 0 to df.shape[0]
            df = df.reset_index(drop=True).reset_index()

            def check_volatility(row, offset):
                idx = row["index"]
                if idx + max_label_length >= df.shape[0]:
                    return np.nan
                else:
                    vol = df.iloc[idx + offset]['close'] / df.iloc[idx]['close'] - 1
                    if vol < -high:
                        return 0
                    elif -high < vol < -low:
                        return 1
                    elif -low < vol < low:
                        return 2
                    elif low < vol < high:
                        return 3
                    elif high < vol:
                        return 4
                    else:
                        return np.nan
            
            df['vol'] = df.apply(lambda row: check_volatility(row, max_label_length), axis=1)
            return df

        def pre_process(df: pd.DataFrame, max_encode_length, max_label_length, is_train = True):
            print("Start preprocessing data")
            df = pd.DataFrame(df)
            exchange_tz = pytz.timezone('Asia/Shanghai')
            df["datetime"] =  df["datetime"].apply(lambda x: datetime.utcfromtimestamp(x.value / 1e9).astimezone(exchange_tz))
            df["is_daytime"] = df["datetime"].apply(lambda x: x.hour * 60 + x.minute)
            train = []
            target = []
            def process_by_group(g) -> pd.DataFrame:        
                g = set_volatility_label(g, max_label_length)
                print(g["vol"].value_counts())
                g = g[self.train_col_name].dropna()
                target_list = []
                train_list = []
                for i in tqdm(range(g.shape[0] - max_encode_length)):
                    # yield training, target 
                    train, target = g.iloc[i:i+max_encode_length].to_numpy(dtype=np.float32), g.iloc[i+max_encode_length]["vol"]
                    train_list.append(train)
                    target_list.append(target)
                return train_list, target_list 
            print("Start processing group")
            for g_name, g in df.groupby("underlying_symbol"):
                print("Processing group: ", g_name)
                if is_train:
                    x, y = process_by_group(g)
                else:
                    x, y = g[self.train_col_name].dropna(), None
                train += x
                target += y
            print("Preprocess data done.")
            return np.array(train), np.array(target)

        return pre_process(data, max_encode_length, max_label_length)

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
        hp = False,
    ) -> Model:
        if hp:
            # hyperparameter tuning
            head_size = hp.Int("head_size", min_value=128, max_value=512, step=64)
            num_heads = hp.Int("num_heads", min_value=1, max_value=6, step=1)
            ff_dim = hp.Int("ff_dim", min_value=2, max_value=8, step=1)
            mlp_dropout = hp.Float("mlp_dropout", min_value=0.1, max_value=0.5, step=0.1)
            dropout = hp.Float("dropout", min_value=0.2, max_value=0.4, step=0.05)

            # mlp_units_key = hp.Choice("mlp_units", values=[0,1,2,3,4])
            # mlp_units = mlp_units_dict[mlp_units_key]
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

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
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
            hp = hp,
        )

        lr = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5]) if hp else 1e-4

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["sparse_categorical_accuracy"],
        )
        model.summary()
        return model
    
    def tune(self):
        """
        Tune hyperparameters
        """
        wandb.init(project="ts_prediction", group="tune")
        tuner = kt.Hyperband(self.model_builder,
                     objective='sparse_categorical_accuracy',
                     max_epochs=5,
                     factor=3,
                     directory='keras_tuner',
                     project_name='ttc_tuner')

        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"), 
            WandbCallback(save_model=False, monitor="sparse_categorical_accuracy", mode="max"),
            tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        ]
        print("Start tuning")

        # tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.get_config())

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


        model = tuner.hypermodel.build(best_hps)

        # history = model.fit(self.X_train, self.y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        hypermodel.fit(self.X_train, self.y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

        eval_result = hypermodel.evaluate(self.X_test, self.y_test)
        print("[test loss, test accuracy]:", eval_result)

    def train(self):
        wandb.init(project="ts_prediction")
        model = self.model_builder()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
            WandbCallback(save_model=True, monitor="sparse_categorical_accuracy", mode="max")
        ]

        model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.25,
            epochs=50,
            batch_size=256,
            shuffle=False, # should not shuffle the data, cuz the data is time series
            callbacks=callbacks,
        )

        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)
        return model

    def predict(self, model, data):
        if isinstance(model, str):
            model: Model = models.load_model(model)
        result = model.predict(data)
        return result

