import numpy as np
import pandas as pd

import random as rn

import sklearn
import xgboost


np.random.seed(42)


train = pd.read_csv("train1.csv",";")
target = pd.read_csv("target1.csv")
test = pd.read_csv("test1.csv",";")

model  = xgboost.XGBRegressor( nrounds = 250, max_depth = 5, eta = 0.05, gamma = 0.08, subsample = 0.96, colsample_bytree = 0.9, min_child_weight = 4, loss = "bernoulli", num_parallel_tree = 200, eval_metric = "logloss")
model.fit(train,target)
#1 max_depth = 5, eta = 3.1, gamma = 0.01, subsample = 0.6, colsample_bytree = 0.7, min_child_weight = 5 - 0.5459864
#(max_depth = 5, eta = 3.5, gamma = 0.01, subsample = 0.7, colsample_bytree = 0.8, min_child_weight = 4) - 5458
#nrounds = 200, max_depth = 5, eta = 0.001, gamma = 0.01, subsample = 0.95, colsample_bytree = 0.7, min_child_weight = 4, num_parallel_tree = 170, loss = "huberized") - 5450
#"huberized"
#"bernoulli"
#
model.fit(train,target)
result = model.predict(test)
result = pd.DataFrame(result)
result.to_csv("r1xgbDsm.csv", index = False)
