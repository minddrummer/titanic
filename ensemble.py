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
from sklearn.preprocessing import StandardScaler as Scaler
from sklearn.svm import SVC
pylab.ion()
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)

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
	def transform(self, X0, y=None):
		X = X0.copy()
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
	def transform(self, X0, y=None):
		X = X0.copy()
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
	def transform(self, X0, y=None, **transform_paras):
		X = X0.copy()
		###the following will change the original dataframe as well
		X.loc[:,'Sex'] = X.Sex.replace({'male':1, 'female':0})
		return X


######name
class ExtractName(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self

	def transform(self, X0, y=None, **transform_paras):
		X = X0.copy()
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
	def __init__(self, ticket_start = False):
		self.ticket_start = ticket_start
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X0, y=None, **transform_paras):
		X = X0.copy()
		tickets = X.Ticket.apply(lambda row: self.extract_letter(row))
		X.loc[:,'Ticket_len'] = tickets.apply(lambda x: len(x))
		if self.ticket_start:
			X.loc[:,'Ticket_start'] = tickets.apply(lambda x: x[0] if len(x)>0 else '0')
		return X
	def extract_letter(self, row):
		# m = re.search('^[A-Za-z]+.* ', row)
		m = re.search('[0-9]+$', row)
		if m: return m.group(0)
		else: return ''


class ProcessTicket2(BaseEstimator, TransformerMixin):
	'''
	this function transfer the ticket number to length(need further categorize to dummy V)
	'''
	def __init__(self, drop):
		self.drop = drop
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X0, y=None, **transform_paras):
		X = X0.copy()
		if self.drop:
			X.drop('Ticket', axis=1, inplace = True)
			return X
		else:
			tickets = X.Ticket.apply(lambda row: row.split(' ')).apply(lambda x: x[0] if len(x)>1 else 'unknown').apply(self.match_string)
			# print tickets
			X.loc[:,'Ticket'] = tickets
			return X
	def match_string(self, row):
		if row in ['A/5','A/5.', 'A./5.', 'A.5.']: 
			return 'A5'
		elif row in ['PC']:
			return 'PC'
		elif row in ['SW/PP', 'S.W./PP', 'PP', 'P/PP', 'S.O./P.P.', 'S.O.P.']:
			return 'PP'
		elif row in ['C.A.', 'CA', 'CA.',  'C.A./SOTON']:
			return 'CA'
		elif row in ['SC/Paris',  'SC/PARIS', 'SC', 'SC/AH', 'S.C./PARIS']:
			return 'SC'
		elif row in ['S.C./A.4.','A/4.',  'A4.', 'A/4']:
			return 'A4'
		elif row in ['S.P.']:
			return 'SP'
		elif row in ['S.O.C.', 'SO/C']:
			return 'SOC'
		elif row in ['W./C.', 'W/C']:
			return 'WC'
		elif row in ['SOTON/OQ' , 'STON/O', 'SOTON/O.Q.', 'STON/O2.' , 'SOTON/O2']:
			return 'SOTON'
		elif row in  ['W.E.P.',  'WE/P']:
			return 'WEP'
		elif row in ['C', 'F.C.C.',  'F.C.']:
			return 'FC'
		else:
			return 'unknown'



#######Fare: no need to feature engineering at this point
#######Cabin
class ProcessCabin(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X0, y = None):
		X = X0.copy()
		cabins = X.Cabin.apply(lambda row: row if type(row) is str else '')
		X.loc[:, 'Cabin'] = cabins.apply(lambda row: self.parse_cabin(row))
		# print X.Cabin
		return X
	def parse_cabin(self, row):
		m = re.search('^[A-Za-z]+?', row)
		if m: 
			if m.group(0) != 'T': return m.group(0)
			else: return 'unknown'
		else: return 'unknown'

class ProcessCabin2(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None, **fit_paras):
		return self
	def transform(self, X0, y = None):
		X = X0.copy()
		X.loc[:, 'Cabin'] = X.Cabin.apply(lambda row: row if type(row) is str else 'unknown')	
		return X
	
#######Embarked
class ProcessEmbarked(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.mode = None
	def fit(self, X, y= None, **fit_paras):
		self.mode = X.Embarked.mode().values[0]
		return self
	def transform(self, X0, y= None, **transform_paras):
		X = X0.copy()
		X.Embarked.fillna(self.mode, inplace = True)
		return X

######Age
##age is important: filling in the missing values(by GLM, or decision tree regression, or other methods)


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

	def transform(self, X0, y = None, **transform_paras):
		X = X0.copy()
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
		age_train = X_train.Age
		X_train = X_train.drop('Age',axis=1)
		self.linearReg.fit(X_train, age_train)			
		# print 'R^2 is', self.linearReg.score(X_train, age_train)	
		# print 'ib predict error is:', (self.linearReg.predict(X_train) - age_train).apply(abs).mean()
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

class ProcessFare(BaseEstimator, TransformerMixin):
	'''
	Fare is not normally distributed, so we transform it to normal shape
	'''
	def __init__(self):
		self.mean = None
		self.std = None
	def fit(self, X, y=None, **fit_paras):
		self.mean = (X.Fare+1).apply(np.log).mean()
		self.std = (X.Fare+1).apply(np.log).std()
		return self
	def transform(self, X0, y=None, **transform_paras):
		X = X0.copy()
		X.loc[:,'Fare'] = ((X.Fare+1).apply(np.log)-self.mean)/self.std
		return X


#############data first segment and drop some variables
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

np.random.seed(119)
select = pd.Series(np.random.random_sample(data.shape[0]), index = data.index)<=0.8
train = data[select].copy()
valid = data[select.apply(lambda x: not x)].copy()

###miscellaneous
train.set_index(['PassengerId'], inplace=True)
valid.set_index(['PassengerId'], inplace=True)
test.set_index(['PassengerId'], inplace=True)
y_train = train.Survived
y_valid = valid.Survived
train.drop('Survived', inplace = True, axis=1)
valid.drop('Survived', inplace = True, axis=1)
test.loc[:,'Fare'].fillna(value = train.loc[:,'Fare'].mean(), inplace=True)
# test_passengerId = test.PassengerId
##combine train and valid for combined fitting after gridsearchCV
df = pd.concat([train, valid], axis=0).copy()


######################################random forest######################################
cross_validation_rf = False
# cross_validation_rf = True

# categorical_vars = ['Pclass','Name','Cabin','Embarked','Ticket_start', 'Ticket']
categorical_vars = ['Pclass','Name','Cabin','Embarked', 'Ticket']

#####pipeline transformation for random forest
# pip_rf=make_pipeline(ProcessFare(), NominalSibSp(), NominalParch(), TransformSex(), ExtractName(), ProcessTicket(), ProcessTicket2(False),\
# 	ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars), LinearRegForAge())

pip_rf=make_pipeline(ProcessFare(), NominalSibSp(), NominalParch(), TransformSex(), \
	ExtractName(), ProcessTicket(ticket_start=False), ProcessTicket2(drop=False),\
	ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars), LinearRegForAge())



if cross_validation_rf:
	print 'first print, data df shape is (%d, %d)' %(train.shape[0], train.shape[1])
	train = pip_rf.fit_transform(train)
	print 'second print, data df shape is (%d, %d)' %(train.shape[0], train.shape[1])
	valid = pip_rf.transform(valid)

	######best parameter setting so far
	# pip_rf=make_pipeline(ProcessFare(), NominalSibSp(), NominalParch(), TransformSex(), ExtractName(), ProcessTicket(), ProcessTicket2(False),\
	# ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars), LinearRegForAge())

# the best parameter setting is: RandomForestClassifier(bootstrap=True, compute_importances=None,
#             criterion='gini', max_depth=None, max_features=9,
#             max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
#             min_samples_split=11, n_estimators=100, n_jobs=1,
#             oob_score=True, random_state=None, verbose=0)
# the best CV score of the GridSearchCV is: 0.844537815126
# the best oob score of the best estimator of grid search results are: 0.830532212885
# the in-bag prediction accuracy rate is: 0.903361344538
# the validation prediction accuracy rate is: 0.819209039548
	######best parameter setting so far

	# # ###random forest classifier
	rfc = RandomForestClassifier(oob_score=True, max_depth=None)
	# #parameter grids: 
	n_estimators = [50, 100, 150, 200]
	max_features = [3, 6, 9, 15,20, 25]
	# max_features = [20,25,30,40]
	min_samples_split = [2, 3, 6, 9, 11, 15]


	#parameter grids: 
	# n_estimators = [125]
	# max_features = [20,25,30]
	# # max_features = [10,20,25,40,60]
	# min_samples_split = [9]


	rf_paras = {'n_estimators': n_estimators, 'max_features':max_features,'min_samples_split':min_samples_split}
	rfc_grid_search = GridSearchCV(rfc, param_grid = rf_paras, cv = 7, refit=True)
	rfc_grid_search.fit(train, y_train)

	y_predict = rfc_grid_search.predict(train)
	y_valid_predict = rfc_grid_search.predict(valid)
	# y_test = rfc_grid_search.predict(test)

	# ###checking results
	print 'the combinations scores of each parameter setting:', sorted(rfc_grid_search.grid_scores_, key=lambda x: x[1])
	print 'the best parameter setting is:', rfc_grid_search.best_estimator_
	print 'the best CV score of the GridSearchCV is:', rfc_grid_search.best_score_
	#best CV score is:
	print 'the best oob score of the best estimator of grid search results are:', rfc_grid_search.best_estimator_.oob_score_
	print 'the in-bag prediction accuracy rate is:', (y_train == y_predict).sum()/float(y_train.shape[0])
	print 'the validation prediction accuracy rate is:', (y_valid == y_valid_predict).sum()/float(y_valid.shape[0])
	# print 'the feature importances are:', sorted(list(zip(train.columns, rfc_grid_search.best_estimator_.feature_importances_)), key = lambda x: x[1], reverse=True)


else:
	# #####combine train and valid to get the final model and predict again
	rfc_self = RandomForestClassifier(oob_score=True, max_depth=None,max_features=9, min_samples_split=11, n_estimators=100)
	# final_df = pd.concat([train, valid], axis = 0)
	final_df = pip_rf.fit_transform(df)
	final_y = pd.concat([y_train, y_valid], axis=0).reset_index().set_index('PassengerId').sort().loc[:,'Survived']
	rfc_self.fit(final_df,final_y)
	print rfc_self.oob_score_
	y_predict = rfc_self.predict(final_df)
	print 'the in-bag prediction accuracy rate of random forest is:', (final_y == y_predict).sum()/float(final_y.shape[0])
	test_rf = pip_rf.transform(test)
	y_test = rfc_self.predict(test_rf)


	# # ###output the data from y_test
	output_rf = pd.Series(y_test, index = test.index).reset_index()
	output_rf.columns = ['PassengerId','Survived']
	# output_rf.to_csv('y_test.csv', index=False)
	## roughly checking performance
	sk = pd.read_csv('y_test_0.csv')
	print (sk.Survived == y_test).sum()/float(sk.Survived.shape[0])


#########################SVM model#########################
# cross_validation_svm = True
cross_validation_svm = False

categorical_vars = ['Pclass','Name','Cabin','Embarked',   'Ticket_start', 'Ticket']
# categorical_vars = ['Pclass','Name','Cabin','Embarked',     'Ticket']

#####pipeline transformation for SVM
# in SVM, you should scale, because the RBF kernel of SVM or the L1 and L2 regularizers of linear models
# assume that all features are centered around 0 and have variance in the same order

# pip_svm=make_pipeline(ProcessFare(), NominalSibSp(), NominalParch(), TransformSex(), \
# 	ExtractName(), ProcessTicket(ticket_start=False), ProcessTicket2(drop=False),\
# 	ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars), LinearRegForAge(), Scaler())

pip_svm=make_pipeline(ProcessFare(), NominalSibSp(), NominalParch(), TransformSex(), \
	ExtractName(), ProcessTicket(ticket_start=True), ProcessTicket2(drop=False),\
	ProcessEmbarked(), ProcessCabin(), CombineDummyVars(categorical_vars), LinearRegForAge())


if cross_validation_svm:
	print 'first print, data df shape is (%d, %d)' %(train.shape[0], train.shape[1])
	train = pip_svm.fit_transform(train)
	print 'second print, data df shape is (%d, %d)' %(train.shape[0], train.shape[1])
	valid = pip_svm.transform(valid)


	######best parameter setting so far
# the best parameter setting is: SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.1,
#   kernel='linear', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# the best CV score of the GridSearchCV is: 0.830532212885
# the in-bag prediction accuracy rate is: 0.841736694678
# the validation prediction accuracy rate is: 0.83050847457
	######best parameter setting so far

	# # ###SVM classifier
	# SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, \
	# 	shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, \
	# 	verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
	svm = SVC()
	# #parameter grids:
	#linear SVM 
	SVM_C = list(np.arange(0.03, 0.16, 0.01))
	SVM_gamma = [0.3]
	# SVM_kernel = ['sigmoid', 'rbf']
	SVM_kernel = ['linear']
	#kernel SVM
	# SVM_C = [0.001, 0.003, 0.1, 0.3, 0.6, 1.0, 3.0, 10, 20]
	# SVM_gamma = [0.001, 0.003, 0.1, 0.3, 0.6, 1.0, 3.0, 10, 20]
	# SVM_kernel = ['sigmoid']


	SVM_paras = {'C': SVM_C, 'gamma':SVM_gamma,'kernel':SVM_kernel}
	SVM_grid_search = GridSearchCV(svm, param_grid = SVM_paras, cv = 7, refit=True)
	SVM_grid_search.fit(train, y_train)
	# SVM_grid_search.fit(train.astype(np.float), y_train.astype(np.float))

	y_predict = SVM_grid_search.predict(train)
	y_valid_predict = SVM_grid_search.predict(valid)

	# ###checking results
	print 'the combinations scores of each parameter setting:'
	for item in sorted(SVM_grid_search.grid_scores_, key=lambda x: x[1]): print item
	print 'the best parameter setting is:', SVM_grid_search.best_estimator_
	print 'the best CV score of the GridSearchCV is:', SVM_grid_search.best_score_
	#best CV score is:
	
	print 'the in-bag prediction accuracy rate is:', (y_train == y_predict).sum()/float(y_train.shape[0])
	print 'the validation prediction accuracy rate is:', (y_valid == y_valid_predict).sum()/float(y_valid.shape[0])
	# print 'the feature importances are:', sorted(list(zip(train.columns, rfc_grid_search.best_estimator_.feature_importances_)), key = lambda x: x[1], reverse=True)


else:
	# #####combine train and valid to get the final model and predict again
	svm_self = SVC(C=0.09, kernel ='linear', gamma= 0.3)
	# svm_self = SVC(C=20 , kernel ='sigmoid', gamma= 0.003)
	# final_df = pd.concat([train, valid], axis = 0)
	final_df = pip_svm.fit_transform(df)
	final_y = pd.concat([y_train, y_valid], axis=0).reset_index().set_index('PassengerId').sort().loc[:,'Survived']
	svm_self.fit(final_df,final_y)
	y_predict = svm_self.predict(final_df)
	print 'the in-bag prediction accuracy rate of SVM is:', (final_y == y_predict).sum()/float(final_y.shape[0])
	test_svm = pip_svm.transform(test)
	y_test = svm_self.predict(test_svm)


	# # ###output the data from y_test
	output_svm = pd.Series(y_test, index = test.index).reset_index()
	output_svm.columns = ['PassengerId','Survived']
	output_svm.to_csv('y_test.csv', index=False)
	## roughly checking performance
	sk = pd.read_csv('y_test_0.csv')
	print (sk.Survived == y_test).sum()/float(sk.Survived.shape[0])



##################Ada-boosting model
