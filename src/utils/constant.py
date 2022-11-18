
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

low_by_label_length = {
    INTERVAL.FIVE_SEC: {
        5: 0.0001, # 0.01%
        10: 0.0005, # 0.05%
    }, 
    INTERVAL.ONE_MIN: {
        10: 0.0015, # 0.15%
        5: 0.00075, # 0.075%
        1: 0.00025, # 0.025%
    }
}

high_by_label_length = {
    INTERVAL.FIVE_SEC: {
        5: 0.0002, # 0.04%
        10: 0.001, # 0.1%
    },
    INTERVAL.ONE_MIN: {
        10: 0.003, # 0.3%
        5: 0.0015, # 0.15%
        1: 0.0005, # 0.05%
    }
}