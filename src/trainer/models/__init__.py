from .tft import TFTModel


class Models:
    def __init__(self, algo_name: str):
        self.algo_name = algo_name

    def get_model(self, data):
        if self.algo_name == "TFT":
            from .tft import TFTModel
            model = TFTModel(data)
        else:
            raise NotImplementedError
        return model