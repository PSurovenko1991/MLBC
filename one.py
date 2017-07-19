import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
import h2o as h2
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.metrics import accuracy_score # Array comparison
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
import math
from sklearn.ensemble import AdaBoostRegressor #!!!

# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import BaggingRegressor
np.random.seed(42)

#LOAD:
data_set = pd.read_csv("train.csv",";")
#!!!
data_set["weight"] = data_set["weight"].apply(lambda x: x+50 if x<31 else x)
data_set["height"] = data_set["height"].apply(lambda x: x+80 if x<100 else x)
data_set["height"] = data_set["height"].apply(lambda x: x-70 if x>200 else x)

# normalization Pressure:
data_set["ap_hi"] = data_set.ap_hi.apply(lambda x: x*(-1) if x <0 else x)
data_set["ap_hi"] = data_set.ap_hi.apply(lambda x: 70 if x < 50 else x)
# data_set["ap_hi"] = data_set.ap_hi.apply(lambda x: x*10 if x < 60 else x)
data_set["ap_hi"] = data_set.ap_hi.apply(lambda x: x/10 if x>241 and x<1000 else x)
data_set["ap_hi"] = data_set.ap_hi.apply(lambda x: x/10 if x>1000 and x<10000 else x)
data_set["ap_hi"] = data_set.ap_hi.apply(lambda x: x/100 if x>10000  else x)
data_set["ap_lo"] = data_set.ap_lo.apply(lambda x: x*(-1) if x <0 else x)
data_set["ap_lo"] = data_set.ap_lo.apply(lambda x: x+100 if x <50 else x)
data_set["ap_lo"] = data_set.ap_lo.apply(lambda x: 80 if x == 0 else x)
data_set["ap_lo"] = data_set.ap_lo.apply(lambda x: x/10 if x >190 else x)
data_set["ap_lo"] = data_set.ap_lo.apply(lambda x: x/100 if x>10000  else x)

data_set["gender"] = data_set["gender"].apply(lambda x: -0.1 if x ==2 else -0.11)
data_set["active"] = data_set["active"]*data_set["gender"]
data_set["active"] = data_set["active"]+1
data_set["cholesterol"] = data_set["cholesterol"]*data_set["active"]

data_set["age"] = data_set["age"].apply(lambda x: (x/365.25))

data_set["age"] = data_set["age"].round(0)
data_set["age>51"] = data_set["age"].apply(lambda x: 1 if x<55 else 1.11)
data_set["ap_hi"] = data_set["ap_hi"]*data_set["age>51"]
data_set["weight"] = data_set["weight"]*data_set["age>51"]
data_set["cholesterol"] = data_set["cholesterol"]*data_set["age>51"]
data_set["gluc"] = data_set["gluc"]*data_set["age>51"]
data_set["index_massi_tela"] =((data_set["height"]))/((data_set["weight"])**(1/2)) # ? wei
data_set["ap_lo"] = (data_set["ap_lo"]**(1/9)) 


#matrix
correlation = data_set.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=1, square=True,  annot=True ) 
plt.show()

fields =  (np.extract(abs((correlation.cardio))>0.1, (correlation.columns)))

fields = np.delete(fields,7) #6
fields = np.delete(fields,6)
fields = np.delete(fields,1) #6
# fields = np.delete(fields,0)

print("Important fields: ",fields)
# print(correlation)



# Save data:
train = data_set[fields]
target = data_set["cardio"]

#Here it was necessary to use cross-validation, in order to avoid over-fit
#(x_train, x_test , y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=0) 

train.to_csv("train1.csv", ";", index=False)
target.to_csv("target1.csv",  index= False)

# & The same for test:
Tdata_set = pd.read_csv("test.csv",";")
# normalization Pressure:
import random
Tdata_set["active"] =Tdata_set.active.apply(lambda x: int(random.randint(0,1)) if x == 'None' else int(x))
Tdata_set["weight"] = Tdata_set["weight"].apply(lambda x: x+70 if x<31 else x)
Tdata_set["height"] = Tdata_set["height"].apply(lambda x: x+100 if x<100 else x)
Tdata_set["height"] = Tdata_set["height"].apply(lambda x: x-100 if x>200 else x)
Tdata_set["ap_hi"] = Tdata_set.ap_hi.apply(lambda x: x*(-1) if x <0 else x)
Tdata_set["ap_hi"] = Tdata_set.ap_hi.apply(lambda x: x/10 if x>241 and x<1000 else x)
Tdata_set["ap_hi"] = Tdata_set.ap_hi.apply(lambda x: x/10 if x>1000 and x<10000 else x)
Tdata_set["ap_hi"] = Tdata_set.ap_hi.apply(lambda x: x/100 if x>10000  else x)
Tdata_set["ap_lo"] = Tdata_set.ap_lo.apply(lambda x: x*(-1) if x <0 else x)
Tdata_set["ap_lo"] = Tdata_set.ap_lo.apply(lambda x: x+100 if x <50 else x)
Tdata_set["ap_lo"] = Tdata_set.ap_lo.apply(lambda x: 80 if x == 0 else x)
Tdata_set["ap_lo"] = Tdata_set.ap_lo.apply(lambda x: x/10 if x >190 else x)
Tdata_set["ap_lo"] = Tdata_set.ap_lo.apply(lambda x: x/100 if x>10000  else x)
Tdata_set["age"] = Tdata_set["age"].apply(lambda x: (x/365))
Tdata_set["age"] = Tdata_set["age"].round(0)
Tdata_set["age>51"] = Tdata_set["age"].apply(lambda x: 1 if x<55 else 1.11)
Tdata_set["ap_hi"] = Tdata_set["ap_hi"]*Tdata_set["age>51"]
Tdata_set["weight"] = Tdata_set["weight"]*Tdata_set["age>51"]
Tdata_set["cholesterol"] = Tdata_set["cholesterol"]*Tdata_set["age>51"]
Tdata_set["gluc"] = Tdata_set["gluc"]*Tdata_set["age>51"]
Tdata_set["index_massi_tela"] =((Tdata_set["height"]))/((Tdata_set["weight"])**(1/2))  # del
Tdata_set["ap_lo"] = (Tdata_set["ap_lo"]**(1/9)) 
Tdata_set["gender"] = Tdata_set["gender"].apply(lambda x: -0.1 if x ==2 else -0.18)
Tdata_set["active"] = Tdata_set["active"]*Tdata_set["gender"]
Tdata_set["active"] = Tdata_set["active"]+1
Tdata_set["cholesterol"] = Tdata_set["cholesterol"]*Tdata_set["active"]

#save:
test = Tdata_set[fields]
test.to_csv("test1.csv", ";", index= False)

#for target1.csv need add first row "cardio"


