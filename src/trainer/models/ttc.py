# Transformer Time series classification 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from datetime import datetime 
from tqdm import tqdm
import pytz
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from wandb.keras import WandbCallback

class TTCModel:
    def __init__(self, data: pd.DataFrame):
        print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
        self.n_classes = 5

        if isinstance(data, pd.DataFrame):
            X, y = self.pre_process_saving(data)
        else:
            X, y = data
        # X = self.timeseries_normalize(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
        self.input_shape = self.X_train.shape[1:]
    
    def timeseries_normalize(self, data: np.ndarray):
        for subset in tqdm(range(data.shape[0])):
            # subset shpae: (max_encode_length, 8)
            scaler = MinMaxScaler(feature_range=(0, 2))
            data[subset] = scaler.fit_transform(data[subset])
            print(data[subset])
        return data
    
    def pre_process_saving(self, data: pd.DataFrame, max_encode_length: int = 60, max_label_length: int = 30):
        def set_volatility_label(df: pd.DataFrame, max_label_length):
            """
                check if the price changes by percentage within max_label_length step
                n_classes = 5
            """

            # reset the index from 0 to df.shape[0]
            df = df.reset_index(drop=True).reset_index()

            def check_volatility(row, offset):
                idx = row["index"]
                if idx + max_label_length >= df.shape[0]:
                    return np.nan
                else:
                    vol = df.iloc[idx + offset]['close'] / df.iloc[idx]['close'] - 1
                    if vol < -0.01:
                        return 0
                    elif -0.01 < vol < -0.005:
                        return 1
                    elif -0.005 < vol < 0.005:
                        return 2
                    elif 0.005 < vol < 0.01:
                        return 3
                    elif 0.01 < vol:
                        return 4
                    else:
                        return np.nan
            
            df['vol'] = df.apply(lambda row: check_volatility(row, max_label_length), axis=1)
            return df

        def pre_process(df: pd.DataFrame, max_encode_length, max_label_length):
            df = pd.DataFrame(df)
            exchange_tz = pytz.timezone('Asia/Shanghai')
            df["datetime"] =  df["datetime"].apply(lambda x: datetime.utcfromtimestamp(x.value / 1e9).astimezone(exchange_tz))
            df["is_daytime"] = df["datetime"].apply(lambda x: x.hour * 60 + x.minute)
            col_name = ["open", "high", "low", "close", "vol", "open_oi", "close_oi", "is_daytime"]
            train = []
            target = []
            def process_by_group(g) -> pd.DataFrame:        
                g = set_volatility_label(g, max_label_length)
                g = g[col_name].dropna()
                target_list = []
                train_list = []
                for i in tqdm(range(g.shape[0] - max_encode_length)):
                    # yield training, target 
                    train, target = g.iloc[i:i+max_encode_length].to_numpy(dtype=np.float32), g.iloc[i+max_encode_length]["vol"]
                    train_list.append(train)
                    target_list.append(target)
                return train_list, target_list 
            print("start processing group")
            for _, g in df.groupby("underlying_symbol"):
                x, y = process_by_group(g)
                train += x
                target += y
            print("Preprocess data done.")
            return np.array(train), np.array(target)
        print("Start preprocessing data")
        return pre_process(data, max_encode_length, max_label_length)

    def build_model(
        self,
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
    ) -> keras.Model:
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
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

    def train(self,):
        wandb.init(project="ts_prediction")

        model = self.build_model(
            self.input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["sparse_categorical_accuracy"],
        )
        model.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
            WandbCallback(save_model=True)
        ]

        model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=256,
            shuffle=True,
            callbacks=callbacks,
        )

        test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)
        return model
