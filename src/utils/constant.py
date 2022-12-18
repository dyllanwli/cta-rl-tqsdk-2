
from utils.utils import Interval
mlp_units_dict = {
    0: [128],
    1: [256],
    2: [512],
    3: [128, 64],
    4: [256, 128],
    5: [512, 256],
    6: [128, 64, 32],
    7: [256, 128, 64],
}

INTERVAL = Interval()

# if n_classes == 3, label range: -inf ~ -low | -low ~ low | low ~ inf
low_by_label_length = {
    INTERVAL.FIVE_SEC: {
        5: 0.0001, # 0.01%
        10: 0.0002, # 0.02%
        20: 0.0005, # 0.05%
    }, 
    INTERVAL.ONE_MIN: {
        1: 0.00025, # 0.025%
        5: 0.00075, # 0.075%
        10: 0.0015, # 0.15%
    }
}

high_by_label_length = {
    INTERVAL.FIVE_SEC: {
        5: 0.0002, # 0.02%
        10: 0.0004, # 0.04%
        20: 0.001, # 0.1%
    },
    INTERVAL.ONE_MIN: {
        1: 0.0005, # 0.05%
        5: 0.0015, # 0.15%
        10: 0.003, # 0.3%
    }
}