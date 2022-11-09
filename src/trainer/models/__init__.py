
class Models:
    def __init__(self, algo_name: str):
        self.algo_name = algo_name

    def get_model(self, data, **kwargs):
        if self.algo_name == "TFT":
            from .tft import TFTModel
            model = TFTModel(data, **kwargs)
        elif self.algo_name == "TTC":
            from .ttc import TTCModel
            model = TTCModel(data, **kwargs)
        else:
            raise NotImplementedError
        return model