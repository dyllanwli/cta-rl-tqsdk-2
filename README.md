# cta-rl-tqsdk-2


## Usage
```
python mt_train.py # train the model
```


## Strategies
+ NW
    - Nadaraya Watson Envelope: for buy and sell signal
    - Nadaraya Watson Estimator: for change in trend
    - RSI + BB + Dispersion (2.0): trend confirmation
    - CPR: for target 
    - Volumes