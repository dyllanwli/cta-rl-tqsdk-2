from datetime import date, datetime

import pandas as pd
import numpy as np
from utils.utils import Interval, max_step_by_day, get_auth
from utils.dataloader import DataLoader, get_symbols_by_names
from .models import Models

from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self, account = "a1", train_type = "tune", max_sample_size = 1e6):
        print("Initializing Model trainer")
        auth = get_auth(account)
        self.train_type = train_type  # tune or train
        self.algo_name = "TTC"
        self.algo = Models(self.algo_name)
        
        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") if self.train_type == "train" else False
        self.project_name = "futures-alpha-8"
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
    
    def get_training_data(self, start_dt=date(2018, 1, 1), end_dt=date(2022, 1, 1)):
        dataloader = DataLoader(start_dt=start_dt, end_dt=end_dt)
        offline_data: pd.DataFrame = dataloader.get_offline_data(
                interval=self.interval, instrument_id=self.symbol, offset=self.max_sample_size, fixed_dt=True)
        return offline_data
            
    
    def run(self):

        # data = self.get_training_data()
        data = [np.load("./X.npy"), np.load("./y.npy")]
        model = self.algo.get_model(data)
        model.train()
        
        # model.print_baseline()
        # model.get_optimal_lr()
        # trainer = model.train()
        # model.test_predict(trainer="/h/diya.li/quant/cta-rl-tqsdk-2/src/lightning_logs/lightning_logs/version_2/checkpoints/epoch=13-step=37058.ckpt")
        # model.tune()
        print("Done")