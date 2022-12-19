import logging 
import os 
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['__MODIN_AUTOIMPORT_PANDAS__'] = '1' # Fix modin warning 

import modin.pandas as pd
import ray
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, models
import keras_tuner as kt
import numpy as np

from tqdm import tqdm
import time

import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

from utils.preprocess import process_prev_close_spread, set_training_label, process_datatime
from utils.tafunc import ema


class TTCModel:
    # Transformer Time series classification 
    def __init__(self, interval: str, commodity_name: str, max_encode_length: int = 60, max_label_length: int = 5):
        ray.init(include_dashboard=False)
        print('GPU name: ', tf.config.list_physical_devices('GPU'))
        self.project_name = "ts_prediction_2"
        self.commodity_name = commodity_name
        self.interval = interval
        self.max_encode_length = max_encode_length
        self.max_label_length = max_label_length

        self.n_classes = 3
        self.train_col_name = ["open", "high", "low", "close", "volume", "is_daytime"] # add moving average will change this
        self.windows = [5, 10, 20, 30, 60]
        self.train_col_name += ["ma_{}".format(window) for window in self.windows]
        self.fit_config = {
            "batch_size": 512,
            "epochs": 100,
            "validation_split": 0.3,
            "shuffle": True,
        }
        self.datatype_name = "{}{}_{}_{}".format(self.commodity_name, self.interval, self.max_encode_length, self.max_label_length)
        self.data_output_path = "./tmp/"+ self.datatype_name+".csv"
        # self.X_output_path = "./tmp/"+ "X_" + self.datatype_name+".npy"
        # self.y_output_path = "./tmp/"+ "y_" + self.datatype_name+".npy"
    
    def _set_classes(self, y: np.ndarray):
        """
        Set the classes for the model
        """
        c = np.array(np.unique(y, return_counts=True)).T
        print("Class distribution: ", c, c[:, 1] / c[:, 1].sum())
        self.n_classes = len(c)
    
    def _add_moving_average(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding moving average")
        for window in self.windows:
            df["ma_{}".format(window)] = ema(df["close"], window)
        return df
    
    def _pre_process_data(self, df: pd.DataFrame, is_prev_close_spread: bool = True):
        # start preprocessing by intervals
        print("Start preprocessing data")
        df = pd.DataFrame(df)
        df = process_datatime(df)

        if is_prev_close_spread:
            df = process_prev_close_spread(df)
        df = set_volatility_label(df, self.max_label_length, self.n_classes, self.interval)
        df = self._add_moving_average(df)
        print(df.shape)
        df = df[self.train_col_name + ["label"]].dropna()
        print(self.train_col_name + ["label"])
        return df
    
    def _set_train_test(self, data: pd.DataFrame):
        X = np.empty(shape = (data.shape[0] - self.max_encode_length, self.max_encode_length, len(self.train_col_name)), dtype=np.float32)
        y = data.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32)
        data = data[self.train_col_name].to_numpy(dtype=np.float32)
        for i in range(X.shape[0]):
            X[i] = data[i:i+self.max_encode_length]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        self._set_classes(y)
        del X, y, data
        self.input_shape = self.X_train[0].shape
        print("Input shape: ", self.input_shape)
    
    def _set_train_dataset(self, data: pd.DataFrame):
        print("Splitting data")
        train, test = train_test_split(data, test_size=0.3, shuffle=False)
        train, val = train_test_split(train, test_size=0.3, shuffle=False)
        
        self._set_classes(train.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32))

        self.train_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=train.iloc[:-self.max_encode_length][self.train_col_name].to_numpy(dtype=np.float32),
            targets=train.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32),
            sequence_length=self.max_encode_length,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.fit_config["batch_size"],
            shuffle=False,
        )
        self.val_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=val.iloc[:-self.max_encode_length][self.train_col_name].to_numpy(dtype=np.float32),
            targets=val.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32),
            sequence_length=self.max_encode_length,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.fit_config["batch_size"]*2,
            shuffle=False,
        )
        self.test_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=test.iloc[:-self.max_encode_length][self.train_col_name].to_numpy(dtype=np.float32),
            targets=test.iloc[self.max_encode_length:]['label'].to_numpy(dtype=np.int32),
            sequence_length=self.max_encode_length,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=self.fit_config["batch_size"],
            shuffle=False,
        )
        self.input_shape = (self.max_encode_length, len(self.train_col_name))

    def set_training_data(self, data: pd.DataFrame):
        if len(data) > 0:
            data = self._pre_process_data(data)
            print("Saving data")
            data.to_csv(self.data_output_path, index=False)
        else:
            data = pd.read_csv(self.data_output_path)

        self._set_train_dataset(data._to_pandas())

    def set_predict_data(self, data: pd.DataFrame):
        print("Set predict data")
        data = self._pre_process_data(data)
        X, y = data[self.train_col_name].to_numpy(), data["label"].to_numpy()
        self._set_classes(y)
        return X, y

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
        feed_forward_type: str = "cnn",
        hp = False,
    ) -> Model:
        if hp:
            # hyperparameter tuning
            head_size = hp.Choice("head_size", values=[128, 256, 512], default=128)
            num_heads = hp.Int("num_heads", min_value=1, max_value=6, step=1)
            ff_dim = hp.Choice("ff_dim", values=[4, 8, 16, 32, 128, 256, 512], default=128)
            mlp_dropout = hp.Float("mlp_dropout", min_value=0.2, max_value=0.4, step=0.1)
            dropout = hp.Float("dropout", min_value=0.2, max_value=0.4, step=0.05)
            num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=3, max_value=8, step=1)
            # mlp_units_key = hp.Choice("mlp_units", values=[64, 128, 256, 512], default=128)
            # mlp_units = hp.Choice("mlp_units", values=[64, 128, 256], default=128)
            lstm_units = hp.Choice("lstm_units", values=[0, 128, 256], default=0)
            feed_forward_type = hp.Choice("feed_forward_type", values=["cnn", "mlp"], default="cnn")
        inputs = keras.Input(shape=input_shape)
        x = inputs
        x = layers.BatchNormalization(epsilon=1e-6)(x)
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout, feed_forward_type)
        
        if lstm_units != 0:
            x = layers.LSTM(lstm_units, return_sequences=True)(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        return Model(inputs, outputs)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0, feed_forward_type="cnn"):
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
        if feed_forward_type == "cnn":
            x = layers.LayerNormalization(epsilon=1e-6)(res)
            x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        elif feed_forward_type == "mlp":
            x = layers.LayerNormalization(epsilon=1e-6)(res)
            x = layers.Dense(ff_dim, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
            x = layers.Dense(inputs.shape[-1])(x)
        return x + res
    
    def model_builder(self, hp = False) -> Model:
        """
        Build the model with hyperparameters
        """
        model_config = dict(
            input_shape = self.input_shape,
            head_size = 512,
            num_heads = 4,
            ff_dim = 4,
            num_transformer_blocks = 4,
            mlp_units = [128, 64],
            dropout = 0.3,
            mlp_dropout = 0.3,
            lstm_units = 0,
            feed_forward_type = "cnn",
        )
        if hp == False:
            print("Model config: ", model_config)
            # update config if not hyperparameter tuning
            wandb.config.update(model_config)

        model = self.build_model(
            input_shape = model_config["input_shape"],
            head_size = model_config["head_size"],
            num_heads = model_config["num_heads"],
            ff_dim = model_config["ff_dim"],
            num_transformer_blocks = model_config["num_transformer_blocks"],
            mlp_units = model_config["mlp_units"],
            dropout = model_config["dropout"],
            mlp_dropout = model_config["mlp_dropout"],
            lstm_units = model_config["lstm_units"],
            feed_forward_type = model_config["feed_forward_type"],
            hp = hp,
        )

        lr = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            decay_rate=0.9,
        )

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
        wandb.init(project=self.project_name, group="tune", reinit=True, settings=wandb.Settings(start_method="fork"), name=self.datatype_name)
        tuner = kt.Hyperband(self.model_builder,
                     objective='sparse_categorical_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='keras_tuner',
                     project_name='ttc_tuner_{}_{}_{}_{}_n{}'.format(self.commodity_name, self.interval, self.max_encode_length, self.max_label_length, self.n_classes))

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"), 
            WandbCallback(save_model=False, monitor="sparse_categorical_accuracy", mode="max"),
            # tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        ]
        print("Start tuning")
        tuner.search(
            self.train_dataset,
            validation_data = self.val_dataset,
            # validation_split=self.fit_config["validation_split"],
            epochs=100, 
            callbacks=callbacks, shuffle=True, batch_size=self.fit_config["batch_size"])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_hps_config = best_hps.get_config()['values']
        print("Best config", best_hps_config)
        # set config to wandb
        wandb.config.update(best_hps_config)

        hypermodel = tuner.hypermodel.build(best_hps)

        # Retrain the model
        hypermodel.fit(
            self.train_dataset,
            validation_data = self.val_dataset,
            # validation_split=self.fit_config["validation_split"],
            epochs=self.fit_config["epochs"],
            batch_size=self.fit_config["batch_size"],
            shuffle=self.fit_config["shuffle"], 
            callbacks=callbacks,
        )

        eval_result = hypermodel.evaluate(self.test_dataset)
        print("[test loss, test accuracy]:", eval_result)

    def train(self):
        wandb.init(project=self.project_name, group="train", reinit=True, settings=wandb.Settings(start_method="fork"), name = self.datatype_name)
        model = self.model_builder()

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True), 
            WandbCallback(save_model=True, monitor="sparse_categorical_accuracy", mode="max")
        ]

        model.fit(
            self.train_dataset,
            validation_data = self.val_dataset,
            # validation_split=self.fit_config["validation_split"],
            epochs=self.fit_config["epochs"],
            batch_size=self.fit_config["batch_size"],
            shuffle=self.fit_config["shuffle"],
            callbacks=callbacks,
        )

        test_loss, test_acc = model.evaluate(self.test_dataset)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_acc)
        return model

    def predict(self, model, X_predict, y):
        if isinstance(model, str):
            model: Model = models.load_model(model)

        # X_predict_norm = self.timeseries_normalize(X_predict)

        eval_result = model.evaluate(X_predict, y)
        print("[predict loss, predict accuracy]:", eval_result)

        print("X_predict shape: ", X_predict.shape)
        for x in X_predict:
            predict = model(np.array([x]), training=False)
            # close_price = X_predict[i, -1, 3]
            # print(predict, close_price, y)
            break
        print(predict)