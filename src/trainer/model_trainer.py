from datetime import date, datetime

import pandas as pd
import numpy as np
from utils.utils import get_auth
from utils.dataloader import DataLoader, get_symbols_by_names
import wandb
from .models import TTCModel
from utils.constant import INTERVAL

class ModelTrainer:
    def __init__(self, account = "a1", train_type = "tune", max_sample_size = 1e8):
        print("Initializing Model trainer")
        auth = get_auth(account)
        self.train_type = train_type  # tune or train
        
        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y%m%d_%H-%M-%S") if self.train_type == "train" else False
        self.project_name = "futures-predict-8"
        self.interval = INTERVAL.ONE_MIN
        self.commodity = "cotton"
        self.symbol = get_symbols_by_names([self.commodity])[0]
        self.max_sample_size = int(max_sample_size)
    
    def get_training_data(self, start_dt=date(2016, 1, 1), end_dt=date(2022, 1, 1)):
        dataloader = DataLoader(start_dt=start_dt, end_dt=end_dt)
        offline_data: pd.DataFrame = dataloader.get_offline_data(
                interval=self.interval, instrument_id=self.symbol, offset=self.max_sample_size, fixed_dt=True)
        return offline_data

    def run(self, is_train=True):
        model = TTCModel(interval=self.interval, commodity_name=self.commodity, max_encode_length=120, max_label_length=1)
        if is_train:
            data = self.get_training_data()
            model.set_training_data(data)
            # model.train()
            # model.tune()
        else:
            predict_data = self.get_training_data(start_dt=date(2022, 1, 1), end_dt=date(2022, 8, 1))
            X_predict, y = model.set_predict_data(predict_data)
            best_model_path = "./tmp/model-best.h5"
            model.predict(best_model_path, X_predict, y)
        
        print("Done")