'''
ensemble SVM, RF and boosting method for titanic, and do powerful feature enigneering
'''
import os
import json
import pandas as pd
import numpy as np
import scipy as sp
import datetime as DT
import pylab
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import FeatureUnion
import matplotlib.pyplot as plt
import re
pylab.ion()
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)


data = pd.read_csv('train.csv')
# PassengerId      int64
# Survived         int64
# Pclass           int64
# Name            object
# Sex             object
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object
# Fare           float64
# Cabin           object
# Embarked        object
# dtype: object

######feature engineering
######SibSp: as int64, for all algorithm
add binary variable based on SibSp?
######Parch: as int64, for all algorithm
######Pclass: nominal varibale, should transform to dummy variable if needed for all algorithm
should we transform to dummy V even it is for RF?
######Sex

######Age



######name



######ticket




#######Fare
#######Cabin
#######Embark


#####for some ML, we need to transfer to dummy variables

