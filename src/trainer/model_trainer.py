from datetime import date, datetime

import pandas as pd
import numpy as np
from utils.utils import Interval, max_step_by_day, get_auth
from utils.dataloader import DataLoader, get_symbols_by_names
import wandb
from .models import TTCModel
class ModelTrainer:
    def __init__(self, account = "a1", train_type = "tune", max_sample_size = 1e7):
        print("Initializing Model trainer")
        auth = get_auth(account)
        self.train_type = train_type  # tune or train
        
        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y%m%d_%H-%M-%S") if self.train_type == "train" else False
        self.project_name = "futures-predict-8"
        INTERVAL = Interval()
        self.interval = INTERVAL.ONE_MIN
        self.max_steps = max_step_by_day[self.interval]
        self.training_iteration = dict({
            INTERVAL.ONE_MIN: 100,
            INTERVAL.FIVE_SEC: 400,
            INTERVAL.ONE_SEC: 500,
        })
        self.symbol = get_symbols_by_names(["cotton"])[0]
        self.max_sample_size = int(max_sample_size)
    
    def get_training_data(self, start_dt=date(2016, 1, 1), end_dt=date(2022, 1, 1)):
        dataloader = DataLoader(start_dt=start_dt, end_dt=end_dt)
        offline_data: pd.DataFrame = dataloader.get_offline_data(
                interval=self.interval, instrument_id=self.symbol, offset=self.max_sample_size, fixed_dt=True)
        return offline_data

    def run(self, is_train=True):
        model = TTCModel()
        if is_train:
            # data = self.get_training_data()
            data = [np.load("./tmp/X_60_5.npy"), np.load("./tmp/y_60_5.npy")]
            model.set_training_data(data, 60, 5)
            # model.train()
            model.tune()
        else:
            predict_data = self.get_training_data(start_dt=date(2022, 1, 1), end_dt=date(2022, 8, 1))
            X_predict, y = model.set_predict_data(predict_data, 90, 10)
            best_model_path = "./tmp/model-best.h5"
            model.predict(best_model_path, X_predict, y)
        
        print("Done")