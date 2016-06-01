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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import statsmodels.formula.api as smf
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
		# X.loc[:,'Name'] = titles_0
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
		if raw == 'Dr.': return 'Mr'
		elif raw == 'Mr.' or raw == 'Master.' or raw == 'Mrs.' or raw == 'Miss.': return re.sub('\.', '', raw)
		elif raw == 'Rev.': return 'Mr'
		elif raw == 'Mlle.': return 'Miss'
		elif raw == 'Major.': return 'Mr'
		elif raw == 'Col.': return 'Mr'
		elif raw == 'Capt.': return 'Mr'
		elif raw == 'Sir.': return 'Mr'
		elif raw == 'Jonkheer.': return 'Master'
		elif raw == 'Jonkvrouw.': return 'Miss'
		elif raw == 'Don.': return 'Mr'
		elif raw == 'the Countess.': return 'Mrs'
		elif raw == 'the Count.': return 'Mr'
		elif raw == 'Ms.': return 'Mrs'
		elif raw == 'Mme.' or raw == 'Madame.': return 'Mrs'
		elif raw == 'Lady.': return 'Mrs'
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
##age is important: filling in the missing values(by GLM, or decision tree regression, or other methods)
class FillingAge(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X, y=None, **transform_paras):
		pass


class CombineDummyVars(BaseEstimator, TransformerMixin):
	'''
	this class create all the dummies for each of the categorical variable in the fit params
	'''
	def __init__(self, var_lst):
		self.create_dummy_dict = {}
		self.var_lst = var_lst
	
	def fit(self, X, y = None, **fit_paras):
		for each in self.var_lst:
			create_dummy = CreateDummy()
			create_dummy.fit(X.loc[:, each])
			self.create_dummy_dict[each] = create_dummy
		return self

	def transform(self, X, y = None, **transform_paras):
		for each in self.var_lst:
			X = pd.concat([X, self.create_dummy_dict[each].transform(X.loc[:, each])], axis=1)
		X.drop(self.var_lst, axis=1, inplace=True)
		return X


class LinearRegForAge(BaseEstimator, TransformerMixin):
	'''
	this class is to fill in the missing values for age,
	since age is very important for prediction
	'''
	def __init__(self):
		# self.linearReg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
		self.linearReg = Ridge(alpha= 0.1, fit_intercept=True, normalize=False, copy_X=True)
		# self.linearReg = Lasso(alpha= 0.5, fit_intercept=True, normalize=False, copy_X=True)
		# self.linearReg = RandomForestRegressor(oob_score=True, n_estimators= 30, max_features='auto')

	def fit(self, X0, y0 = None, **fit_paras):
		'''
		use X0's all not np.NaN data as the training data
		'''
		X_train = X0[~X0.Age.apply(np.isnan)].copy()
		# np.random.seed(199)
		# select = pd.Series(np.random.random_sample(X_train.shape[0])<=0.7, index = X_train.index).apply(lambda x: True if x==1 else False)
		# print select
		# X_test = X_train[select.apply(lambda x: not x)]
		# X_train = X_train[select]		
		# print X_test.shape
		age_train = X_train.Age
		# age_test = X_test.Age
		X_train = X_train.drop('Age',axis=1)
		# X_test = X_test.drop('Age',axis=1)
		# X_train.loc[:,'Fare']=  X_train.Fare.apply(lambda x: np.log(x) if x != 0 else np.log(x+0.001))
		# X_train.loc[:,'Fare'] = (X_train.loc[:,'Fare'] - X_train.loc[:,'Fare'].mean())/X_train.loc[:,'Fare'].std()
		self.linearReg.fit(X_train, age_train)	
		
		print 'R^2 is', self.linearReg.score(X_train, age_train)	
		# print 'R^2 is for test', self.linearReg.score(X_test, age_test)	
		print 'ib predict error is:', (self.linearReg.predict(X_train) - age_train).apply(abs).mean()
		# print 'oob predict error is:', (self.linearReg.predict(X_test) - age_test).apply(abs).mean()
		# print X_train.loc[:,'Fare'].mean()

		# print X_train.head(20)
		# self.linearReg = smf.ols(formula = 'Age~1+Name_Mr+Name_Mrs+Name_Miss+Name_Master+Fare+Sex+Pclass_1+Pclass_2+Pclass_3+Pclass_1:Fare+Pclass_2:Fare+Pclass_3:Fare', data = X_train)
		# self.linearReg = smf.ols(formula = 'Age~Name+Fare^2+Sex+Pclass+Pclass:Fare-1', data = X_train)

		# results = self.linearReg.fit()
		# print results.summary()
		# print 'the linear Regression R^2 is:', self.linearReg.score(X_train, age_train)
		return self

	def transform(self, X, y = None, **transform_paras):
		'''
		for any X0, pull all the np.NaN for age data, and run the prediction for them,
		and save them back to X0, and return X0
		'''
		X0 = X.copy()
		# data with age all is np.nan
		X_test = X0[X0.Age.apply(np.isnan)].drop('Age',axis=1)
		#the original training dataset
		X_train = X0[~X0.Age.apply(np.isnan)]
		X_test.loc[:,'Age'] = self.linearReg.predict(X_test)
		return pd.concat([X_train, X_test], axis=0).sort()



data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

###miscellaneous
y = data.Survived
data.drop('Survived', inplace = True, axis=1)
test.loc[:,'Fare'].fillna(value = data.loc[:,'Fare'].mean(), inplace=True)
test_passengerId = test.PassengerId
test.drop(['PassengerId'], axis=1, inplace=True)
data.drop(['PassengerId'], axis=1, inplace=True)


# ###pipeline transformation---test lots of pipeline combinations
categorical_vars = ['Pclass','Name','Cabin','Embarked']

# # pip1=make_pipeline(NominalSibSp(), NominalParch(), TransformSex(), ExtractName(),ProcessTicket(), \
# # 	ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars), LinearRegForAge())

pip1=make_pipeline(NominalSibSp(), NominalParch(), TransformSex(), ExtractName(),ProcessTicket(), \
	ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars))



print 'first print, data df shape is (%d, %d)' %(data.shape[0], data.shape[1])
data = pip1.fit_transform(data)
print 'second print, data df shape is (%d, %d)' %(data.shape[0], data.shape[1])
test = pip1.transform(test)


linear_reg_for_age = LinearRegForAge()
linear_reg_for_age.fit(data)







# ###random forest classifier
# rfc = RandomForestClassifier(oob_score=True, max_depth=None)
# #parameter grids
# n_estimators = [100, 125, 150, 175]
# max_features = [3,6,10,15,20,25]
# min_samples_split = [2,5,7,9]
# rf_paras = {'n_estimators': n_estimators, 'max_features':max_features,'min_samples_split':min_samples_split}

# rfc_grid_search = GridSearchCV(rfc, param_grid = rf_paras, cv = 10, refit=True)
# rfc_grid_search.fit(data, y)

# y_predict = rfc_grid_search.predict(data)
# y_test = rfc_grid_search.predict(test)

# ###checking results
# print 'the combinations scores of each parameter setting:', sorted(rfc_grid_search.grid_scores_, key=lambda x: x[1])
# print 'the best parameter setting is:', rfc_grid_search.best_estimator_
# print 'the best CV score of the GridSearchCV is:', rfc_grid_search.best_score_
# #best CV score is:
# print 'the best oob score of the best estimator of grid search results are:', rfc_grid_search.best_estimator_.oob_score_
# print 'the in-bag prediction accuracy rate is:', (y == y_predict).sum()/float(y.shape[0])
# print 'the feature importances are:', sorted(list(zip(data.columns, rfc_grid_search.best_estimator_.feature_importances_)), key = lambda x: x[1], reverse=True)


# # ####self setting of rfc
# # rfc_self = RandomForestClassifier(oob_score=True, max_depth=None,max_features=6, min_samples_split=9, n_estimators=100)
# # rfc_self.fit(data,y)
# # print rfc_self.oob_score_
# # y_predict = rfc_self.predict(data)
# # print 'the in-bag prediction accuracy rate is:', (y == y_predict).sum()/float(y.shape[0])
# # y_test = rfc_self.predict(test)

# ###output the data from y_test
# output = pd.concat([test_passengerId, pd.Series(y_test)], axis=1)
# output.columns = ['PassengerId','Survived']
# output.to_csv('y_test.csv', index=False)

# #random forest model or similar for age prediction
# #other techniques from forum
# #ensembling
