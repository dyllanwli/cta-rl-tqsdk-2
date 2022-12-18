import os 
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pytz
import pickle

from matplotlib import pyplot as plt
import numpy as np
import modin.pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
# from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters as tft_tune

from pytorch_forecasting.data.examples import get_stallion_data

from tqdm import tqdm
from collections import defaultdict

# fix bug pytorch/30966
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class TFTModel:
    """
    # model.print_baseline()
    # model.get_optimal_lr()
    # trainer = model.train()
    # model.test_predict(trainer="/h/diya.li/quant/cta-rl-tqsdk-2/src/lightning_logs/lightning_logs/version_2/checkpoints/epoch=13-step=37058.ckpt")
    # model.tune()
    """
    def __init__(self,
        data: pd.DataFrame,
        batch_size: int = 128,
        tune_result_name: str = "tft_tuned"):
        
        self.max_prediction_length = 2
        self.max_encoder_length = 60

        self.tuned_result_name = tune_result_name
        data = self._pre_process(data)
        self.training, self.validation = self._build_dataloader(data)
        self.train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
        self.val_dataloader = self.validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        # configure network and trainer
        pl.seed_everything(42)

    def _pre_process(self, data: pd.DataFrame):
        print("Preprocessing data...")
        data = pd.DataFrame(data)
        exchange_tz = pytz.timezone('Asia/Shanghai')
        data["datetime"] =  data["datetime"].apply(lambda x: datetime.utcfromtimestamp(x.value / 1e9).astimezone(exchange_tz))
        data["time_idx"] = data.index.values
        data["is_daytime"] = data["datetime"].apply(lambda x: "1" if (x.hour >= 8 and x.hour < 16) else "0")
        print("Preprocess data done.")
        return data._to_pandas()

    def _build_dataloader(self, data: pd.DataFrame):
        max_prediction_length = self.max_prediction_length
        max_encoder_length = self.max_encoder_length
        training_cutoff = data["time_idx"].iloc[-1] - max_prediction_length
        # variable_groups = {"time_variable": ["is_daytime"],}

        training: TimeSeriesDataSet = TimeSeriesDataSet(
            data[lambda x: x["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="close",
            group_ids=["underlying_symbol"],
            weight=None,
            min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["underlying_symbol"],
            static_reals=["duration"],
            time_varying_known_categoricals=["is_daytime"],
            # variable_groups=variable_groups, 
            # time_varying_known_reals=["time_idx"],
            # time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_oi",
                "close_oi",
            ],
            # target_normalizer=GroupNormalizer(groups=["instrument_id", "duration"], transformation="softplus"),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            categorical_encoders={"underlying_symbol": NaNLabelEncoder(add_nan=True)},
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        validation: TimeSeriesDataSet = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
        return training, validation

    def train(self):
        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="gpu",
            devices=-1, 
            enable_model_summary=True,
            gradient_clip_val=0.1,
            # limit_train_batches=100,  # comment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.01,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,

            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        # fit network
        trainer.fit(
            tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
        return trainer

    def tune(self):
        # create study
        study = tft_tune(
            self.train_dataloader,
            self.val_dataloader,
            model_path="optuna_test",
            n_trials=100,
            max_epochs=30,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 256),
            hidden_continuous_size_range=(8, 256),
            attention_head_size_range=(1, 4),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(
                accelerator="gpu",
                devices=[0], 
                limit_train_batches=30,
                ),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
            # verbose=True,
        )

        # save study results - also we can resume tuning at a later point in time
        with open("checkpoints/{}.pkl".format(self.tuned_result_name), "wb") as fout:
            pickle.dump(study, fout)

        # show best hyperparameters
        print(study.best_trial.params)
    
    def predict(self, trainer, data):
        """
        new_data = self.get_training_data(start_dt=date(2022, 7, 1), end_dt=date(2022, 8, 1))
        model.predict(trainer="/h/diya.li/quant/cta-rl-tqsdk-2/src/lightning_logs/lightning_logs/version_2/checkpoints/epoch=13-step=37058.ckpt", data = new_data)
        """
        if isinstance(trainer, str):
            model: TemporalFusionTransformer = TemporalFusionTransformer.load_from_checkpoint(trainer)
        else:
            model: TemporalFusionTransformer = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        data = self._pre_process(data)
        actions = 0
        actuals = 0
        predicted_true = 0

        for i in tqdm(range(data.shape[0] - self.max_encoder_length - self.max_prediction_length - 1)): 
            subset = data.iloc[i:i+self.max_encoder_length]

            next_price = data.iloc[i+self.max_encoder_length+self.max_prediction_length]["close"]
            price = subset.iloc[-1]["close"]
            try:    
                _, validation = self._build_dataloader(subset)
                val_dataloader = validation.to_dataloader(train=False, batch_size=1, num_workers=0)

                predictions = model.predict(val_dataloader)
                threshold = 0.01

                if next_price > price:
                    actual_action = 2
                elif next_price == price:
                    actual_action = 1
                else:
                    actual_action = 0

                if price - predictions[0][1] > threshold:
                    # long
                    action = 2
                elif price - predictions[0][1] < - threshold:
                    # short
                    action = 0
                else:
                    # hold
                    action = 1
                if action == actual_action or action == 1:
                    predicted_true += 1
                actions += action
                actuals += actual_action
            except KeyError as e:
                continue
        print(actions, actuals, "win_rate:", predicted_true/actions)

    def test_predict(self, trainer = None):
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        if isinstance(trainer, str):
            model: TemporalFusionTransformer = TemporalFusionTransformer.load_from_checkpoint(trainer)
        else:
            model: TemporalFusionTransformer = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        # calcualte mean absolute error on validation set
        actuals = torch.cat([y[0] for x, y in iter(self.val_dataloader)])
        predictions = model.predict(self.val_dataloader)
        mean_error = (actuals - predictions).abs().mean()
        print("mean_error", mean_error)
        print("Plotting predictions")
        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        raw_predictions, x = model.predict(self.val_dataloader, mode="raw", return_x=True, show_progress_bar=True)

        for idx in range(len(actuals[:10])):  
            fig: plt.Figure = model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
            fig.set_size_inches(15,15)
            plt.savefig("figures/prediction_{}.png".format(idx))
            plt.close(fig)
        
        print("Plotting worst predictions")
        # calcualte metric by which to display
        predictions = model.predict(self.val_dataloader)
        mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
        indices = mean_losses.argsort(descending=True)  # sort losses
        for idx in range(len(actuals[:10])): 
            model.plot_prediction(
                x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=model.loss.quantiles)
            )
            plt.savefig("figures/bad_prediction_{}.png".format(idx))
            plt.close(fig)
        # plot worst 10 examples
        predictions, x = model.predict(self.val_dataloader, return_x=True)
        predictions_vs_actuals = model.calculate_prediction_actual_by_variable(x, predictions)
        fig = model.plot_prediction_actual_by_variable(predictions_vs_actuals)
        plt.savefig("figures/variable_prediction_{}.png".format(idx))
        plt.close()

    
    def print_baseline(self):
        # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
        actuals = torch.cat([y for x, (y, weight) in iter(self.val_dataloader)])
        baseline_predictions = Baseline().predict(self.val_dataloader)
        baseline_error = (actuals - baseline_predictions).abs().mean().item()
        print(f"Baseline MAE is {baseline_error}")
        return baseline_error

    def get_optimal_lr(self):

        lr_finder = pl.Trainer(
            accelerator="gpu",
            devices=-1,
            # clipping gradients is a hyperparameter and important to prevent divergance
            # of the gradient for recurrent neural networks
            gradient_clip_val=0.1,
        )
        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.03,
            hidden_size=16,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=8,  # set to <= hidden_size
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
        # find optimal learning rate
        res = lr_finder.tuner.lr_find(
            tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )
        print(f"suggested learning rate: {res.suggestion()}")
        fig: plt.Figure = res.plot(show=True, suggest=True)
        plt.savefig("figures/lr_find.png")
        plt.close(fig)