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
# add binary variable based on SibSp 
#binary data no need for dummy variables
class NominalSibSp(BaseEstimator, TransformerMixin):
	'''this class divided the SibSp into different class by the mean'''
	def __init__(self):
		self.mean = np.NaN
	def fit(self, X, y=None):
		self.mean = X.loc[:,'SibSp'].mean()
		return self
	def transform(self, X, y=None):
		X.loc[:, 'SibSp_binary'] = (X.loc[:,'SibSp']>=self.mean).astype(int)
		return X

######Parch: as int64, for all algorithm
# add binary variable based on Parch  
#binary data no need for dummy variables
class NominalParch(BaseEstimator, TransformerMixin):
	'''this class divided the SibSp into different class by the mean'''
	def __init__(self):
		self.mean = np.NaN
	def fit(self, X, y=None):
		self.mean = X.loc[:,'Parch'].mean()
		return self
	def transform(self, X, y=None):
		X.loc[:, 'Parch_binary'] = (X.loc[:,'Parch']>=self.mean).astype(int)
		return X

######Pclass: nominal varibale, should transform to dummy variable if needed for all algorithm
# should we transform to dummy V even it is for RandomForestModel? Yes, for convenience
# and more importantly, should based on training, not testing: how to achieve it?
class CreateDummy(BaseEstimator, TransformerMixin):
	'''
	this class create dummy variables for any nominal variable in pandas Dataframe,
	and when transform the new data, if any label is not known, it will be classified as all Os
	it will create all variables corresponding to all levels in pd.DataFrame framework;
	this class doesnot handle missing value: the best is to fill in before use this class; if not,
	it will create NaN binary variable, but the binary name is not proper
	'''
	def __init__(self):
		self.binary_labels = None
		self.root_name = None

	def fit(self, X, y = None):
		self.binary_labels = list(X.unique())
		self.root_name = X.name
		self.var_names = [self.root_name+'_'+ str(binary_label) for binary_label in self.binary_labels]
		return self
	def transform(self, X, y = None):
		# print self.var_names
		res = pd.DataFrame(columns = self.var_names, index = X.index)
		res.loc[:,:] = 0
		# print res
		res.loc[:, 'VarName'] = X.apply(lambda x: self.root_name+'_'+str(x))
		# print res	
		def match_binary_var_be_0(row):
			###make any variable showing in raw.index as 1; if there is any unseen variable,
			###any of these variables would be shown as all 0s in the dummy Variables
			if row.VarName in row.index:
				row.loc[row.VarName] = 1
			return row
		res = res.apply(match_binary_var_be_0, axis=1)
		res.drop('VarName', axis=1, inplace=True)
		return res

create_dummy = CreateDummy()
create_dummy.fit(data.Pclass)
create_dummy.transform(data.Pclass)
# create_dummy.transform(pd.Series([0,1,2,3,4],index = [0,1,2,3,4]))
#we can transform Pclass into binary variables now


######Sex
class TransformSex(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X, y=None, **transform_paras):
		###the following will change the original dataframe as well
		X.loc[:,'Sex'] = X.Sex.replace({'male':1, 'female':0})
		return X


######name
class ExtractName(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self

	def transform(self, X, y=None, **transform_paras):
		titles_0 = X.Name.apply(lambda row: self.parse(row))
		X.loc[:,'Name'] = titles_0.apply(lambda cell: self.transfer_title(cell))
		return X

	def transfer_title(self, raw):
		'''
		replace all the small/rare title to usual ones;
		Mr. Mrs. Miss. Master. should be kept, others should be transfered to these four;
		Master is replaced by Mister(Mr.) in 19th century, titanic happned in 1912, so master. meaning boys and young men;
		Major. -> Mr.
		Mlle. -> Miss.
		Ms    -> Mrs.
		Mme. Madame-> Mrs.
		Rev.-> Mr.
		Dr. -> Mr.
		'''
		if raw == 'Dr.': return 'Mr.'
		elif raw == 'Mr.' or raw == 'Master.' or raw == 'Mrs.' or raw == 'Miss.': return raw
		elif raw == 'Rev.': return 'Mr.'
		elif raw == 'Mlle.': return 'Miss.'
		elif raw == 'Major.': return 'Mr.'
		elif raw == 'Col.': return 'Mr.'
		elif raw == 'Capt.': return 'Mr.'
		elif raw == 'Sir.': return 'Mr.'
		elif raw == 'Jonkheer.': return 'Master.'
		elif raw == 'Jonkvrouw.': return 'Miss.'
		elif raw == 'Don.': return 'Mr.'
		elif raw == 'the Countess.': return 'Mrs.'
		elif raw == 'the Count.': return 'Mr.'
		elif raw == 'Ms.': return 'Mrs.'
		elif raw == 'Mme.' or raw == 'Madame.': return 'Mrs.'
		elif raw == 'Lady.': return 'Mrs.'
		else: return 'Unknown'

	def parse(self, cell):
		m = re.search(', [A-Za-z\W]+?\.', cell) #this way we got all the parsing right, but may get some results that are rare
		# m = re.search(', [A-Za-z]+\.', cell) #this way is more conservative, but would make the rare ones into '' and make them easier to handle
		if m: return m.group(0)[2:]
		else: return ''


######ticket
class ProcessTicket(BaseEstimator, TransformerMixin):
	'''
	this function transfer the ticket number to length(need further categorize to dummy V)
	'''
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X, y=None, **transform_paras):
		tickets = X.Ticket.apply(lambda row: self.extract_letter(row))
		X.loc[:,'Ticket_len'] = tickets.apply(lambda x: len(x))
		X.drop('Ticket', axis=1, inplace=True)
		return X
	def extract_letter(self, row):
		# m = re.search('^[A-Za-z]+.* ', row)
		m = re.search('[0-9]+$', row)
		if m: return m.group(0)
		else: return ''

#######Fare: no need to feature engineering at this point
#######Cabin
class ProcessCabin(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X, y = None):
		cabins = X.Cabin.apply(lambda row: row if type(row) is str else '')
		X.loc[:, 'Cabin'] = cabins.apply(lambda row: self.parse_cabin(row))
		return X
	def parse_cabin(self, row):
		m = re.search('^[A-Za-z]+?', row)
		if m: return m.group(0)
		else: return 'unknown'

#######Embarked
class ProcessEmbarked(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.mode = None
	def fit(self, X, y= None, **fit_paras):
		self.mode = X.Embarked.mode().values[0]
		return self
	def transform(self, X, y= None, **transform_paras):
		X.Embarked.fillna(self.mode, inplace = True)
		return X


######Age
##age is important? build a model to predict age?


# print 'Training...'
# print 'Predicting...'
# print 'Done.'




