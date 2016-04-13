'''
script to pull all the data, processed and put them in a dataframe
'''
import os
import json
import pandas as pd
import numpy as np
import scipy as sp
import combine_connection as conn
import datetime as DT
import pylab
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import FeatureUnion

pylab.ion()
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)


def fetch_and_process_raw_reports_data():
	#count = 0
	report_lst = []
	for line in CLIENT.reports.reports.find():
		# count += 1
		line_dict = line.copy()
		##processing the 'fired'
		if 'fired' not in line_dict:
			line_dict['fired'] = []
		elif (line_dict['fired']) == None:
			line_dict['fired'] = []
		elif len(line_dict['fired']) == 0:
			line_dict['fired'] = []
		elif len(line_dict['fired']) == 1:
			line_dict['fired'] = line_dict['fired']
		else:
			line_dict['fired'] = line_dict['fired']

		##processing the 'food'
		if 'food' not in line_dict:
			line_dict['food'] = None
		elif (line_dict['food']) == None:
			pass
		else:
			for key in (line_dict['food']):
				line_dict['food_'+key] = line_dict['food'][key]

		##processing 'average'
		# if 'average' not in line_dict:
		# 	line_dict['average'] = None
		# elif (line_dict['average']) == None:
		# 	pass
		# else:
		# 	for key in (line_dict['average']):
		# 		line_dict['average_'+key] = line_dict['average'][key]

		##processing 'body'
		if 'body' not in line_dict:
			line_dict['body'] = None
		elif (line_dict['body']) == None:
			pass
		else:
			for key in (line_dict['body']):
				line_dict['body_'+key] = line_dict['body'][key]

		##processing 'sleep'
		if 'sleep' not in line_dict:
			line_dict['sleep'] = None
		elif (line_dict['sleep']) == None:
			pass
		else:
			for key in (line_dict['sleep']):
				line_dict['sleep_'+key] = line_dict['sleep'][key]

		##processing 'activity'
		if 'activity' not in line_dict:
			line_dict['activity'] = None
		elif (line_dict['activity']) == None:
			pass
		else:
			for key in (line_dict['activity']):
				line_dict['activity_'+key] = line_dict['activity'][key]

		#printLog(len(line_dict))
		report_lst.append(line_dict)

	report_df = pd.DataFrame(report_lst)
	#report_df.rename(columns = {'_id': 'reports_id'}, inplace = True)
	#print('the current shape of report_df is:', report_df.shape)
	report_df.drop(['_id','components','projects','activity','average',\
		'clusterValue','food','sleep','body', 'projectId', 'timestamp', 'activity_updated',\
		'fired', 'date', 'range','count','sleep_start','activity_floors'], axis=1, inplace = True)
	#print('the current shape of report_df is:', report_df.shape)
	return report_df

def fetch_and_process_profile_data():
	'''
	fetch and process the profiles data
	'''
	cursor = CLIENT.profiles.profiles.find()
	lst = []
	for line in cursor:
		lst.append(line)

	prof_df = pd.DataFrame(lst)
	prof_df.drop(['_id','avatar','city','state','country','home','tzone','tzoneOffset'], axis =1 ,inplace=True)
	#print prof_df.shape
	return prof_df

def compute_age(df):
	NOW = DT.datetime.now()
	# print NOW
	age = (NOW - pd.to_datetime(df['birthday'])).astype('timedelta64[Y]')
	#filtering on age: is age is less than 5, or age is bigger than 100, make it as np.NaN
	#print age.min()
	age[age<=5.0] = np.NaN
	#print age.min()
	age[age>=100.0] = np.NaN
	df.loc[:,'age'] = age

def extract_device_info(df):
	df.loc[:,'mobile_platform'] = df.device.apply(lambda x: x[u'platform'] if x is not None else None)

def extract_trackers_row_platform(row):
	'''only return the first element value'''
	if len(row) == 0:
		return None
	else:
		return row[0]['platform']

def extract_trackers_row_type(row):
	'''only return the first element value'''
	if len(row) == 0:
		return None
	else:
		return row[0]['type']

def clean_prof_df(df):
	df.loc[:, 'trackers_type'] = df.trackers.apply(extract_trackers_row_type)		
	df.loc[:, 'trackers_platform'] = df.trackers.apply(extract_trackers_row_platform)		
	extract_device_info(df)
	df.drop(['birthday','trackers','device'], axis =1 ,inplace = True)
		
#process the df, to simply take the mean first
#want to try withthis or not withthis?
def filter_credId_by_num_cases(df, num_thres = 5):
	max_cases = df.groupby('credId').count().apply(max, axis = 1)
	#the following are credId that are less than 5 datapoints
	return list(max_cases[max_cases <= num_thres].index)

def fillin_noneand0_asNaN(df, var):
	df.loc[:,var] = df.loc[:,var].apply(lambda x: np.NaN if x is None or x ==0  else x)

def fillin_none_asNaN(df, var):
	df.loc[:,var] = df.loc[:,var].apply(lambda x: np.NaN if x is None else x)

def fillin_all_float_variables(df):
	'''fill in all float None or 0 as np.NaN'''
	var_list = [u'activity_activeMinutes',
	 u'activity_calories',
	 u'activity_distance',
	 u'activity_nonactiveMinutes',
	 u'activity_steps',
	 u'body_bmi',
	 u'body_bodyFat',
	 u'body_weight',
	 u'food_calories',
	 u'food_carbs',
	 u'food_fat',
	 u'food_fiber',
	 u'food_protein',
	 u'food_sodium',
	 u'food_water',
	 u'sleep_asleep',
	 u'sleep_duration']
	for var in var_list: 
		fillin_noneand0_asNaN(df, var)	

def check_data_regularity_float(df,var):
	print 'the df is:', df.shape
	#could also use pd.fillna function actually
	#print the total number cases of when x is None
	print df.loc[:,var].apply(lambda x: True if x is None else False).sum()
	#print the total number cases of when x is np.NaN
	print df.loc[:,var].apply(lambda x: True if np.isnan(x) else False).sum()

def print_all_float_var_checking():
	var_lst = [u'activity_activeMinutes',
	 u'activity_calories',
	 u'activity_distance',
	 u'activity_nonactiveMinutes',
	 u'activity_steps',
	 u'body_bmi',
	 u'body_bodyFat',
	 u'body_weight',
	 u'food_calories',
	 u'food_carbs',
	 u'food_fat',
	 u'food_fiber',
	 u'food_protein',
	 u'food_sodium',
	 u'food_water',
	 u'sleep_asleep',
	 u'sleep_awake',
	 u'sleep_duration',
	 u'height',
	 u'weight',
	 u'age',
	 u'case_thres']
	for var in var_lst:
		print '#############################'
		print 'the current variable is %s' %var 
		check_data_regularity_float(fdf, var)

def fillin_string_missing_value(row):
	'''to string type missing value:
	make missing value or non-sense value as None
	'''
	if type(row) is str or type(row) is unicode:
		if len(row) == 0:
			return None
		else:
			return row
	elif row is None:
		return None
	else:
		return  None	

############################pre-process the data############################
CLIENT = conn.mongo
df  = fetch_and_process_raw_reports_data()
prof_df = fetch_and_process_profile_data()
compute_age(prof_df)
clean_prof_df(prof_df)
prof_df.set_index('credId', inplace = True)

#get all credId that have <= 5 records
unqualified_credIds = filter_credId_by_num_cases(df, num_thres = 5)
fillin_all_float_variables(df)
# for u'sleep_awake', just None, No 0
fillin_none_asNaN(df, 'sleep_awake')


#take the mean
mean_df = df.groupby('credId').mean()
mean_df.loc[:, 'platform_type'] = ''
#for type, take the first
for each_gp in df.loc[:,['credId','type']].groupby('credId'):
	mean_df.loc[each_gp[0], 'platform_type'] = dict(each_gp[1].type.value_counts()).keys()[0]

#combine mean_df with prof_df
fdf = pd.concat([mean_df, prof_df], axis = 1)



#add indicator for credId if the records are less than the num_thres
fdf.loc[:,'case_thres'] = 1
for credId in unqualified_credIds:
	fdf.loc[credId,'case_thres'] = 0

# print_all_float_var_checking()
#first for all string type variables: fill in all missing value as None instead of np.NaN
string_var_lst = [u'platform_type',
 u'gender',
 u'work',
 u'trackers_type',
 u'trackers_platform',
 u'mobile_platform']
for var in string_var_lst:
	fdf.loc[:,var] =  fdf.loc[:,var].apply(fillin_string_missing_value)

#match 'trackers_platform' and 'platform_type'
def generate_platform_info(row):
	if row.platform_type is not None:
		return row.platform_type
	else:
		if row.trackers_platform is not None: 
			return row.trackers_platform
		else:
			return None
fdf.loc[:,'platform'] = fdf.apply(generate_platform_info, axis = 1)
fdf.drop(['trackers_platform','platform_type','work'], axis =1 , inplace =True)


#clean gender
def transform_gender_to_float(cell):
	'''
	encode male = 1; female = 0
	'''
	if cell is None:
		return np.NaN
	elif cell.lower() == 'male' or cell.lower() == 'm':
		return 1
	else:
		return 0
fdf.loc[:,'gender'] = fdf.gender.apply(transform_gender_to_float)

#rename some features here
fdf.loc[:,'platforms'] = fdf.loc[:,'trackers_type']
fdf.loc[:,'mobiles'] = fdf.loc[:,'mobile_platform']
fdf.loc[:,'trackers'] = fdf.loc[:,'platform']
fdf.drop(['trackers_type','mobile_platform','platform'],axis=1, inplace =True)

#rebuild some feature such as weight, bmi etc;
#weight from lb to kg
fdf.loc[:,'weight'] = fdf.loc[:,'weight']*0.454
#height from inch to meters
fdf.loc[:,'height'] = fdf.loc[:,'height']*0.0254

fdf.loc[:,'weight'] = fdf.loc[:,'weight'].apply(lambda x: np.NaN if x<=10 or x>=300 else x)
fdf.loc[:,'body_weight'] = fdf.loc[:,'body_weight'].apply(lambda x: np.NaN if x<=10 or x>= 800 else x)
fdf.loc[:,'body_weight'] = fdf.loc[:,'body_weight'].apply(lambda x: x*0.454 if x>=250 else x)

fdf.loc[:,'height'] = fdf.loc[:,'height'].apply(lambda x: np.NaN if x<=0.3 or x>= 3.0 else x)
fdf.loc[:,'body_bmi'] = fdf.loc[:,'body_bmi'].apply(lambda x: np.NaN if x<=3 or x>= 70 else x)

fdf.loc[:, 'weight'] = fdf.apply(lambda x: x.loc['body_weight'] if np.isnan(x.loc['weight']) else x.loc['weight'] , axis = 1)
fdf.drop(['body_weight','body_bmi'],axis=1,inplace=True)
fdf.loc[:,'bmi'] = fdf.loc[:,'weight']/np.power(fdf.loc[:,'height'], 2)
# fdf.loc[:,['bmi','height','weight']]


class RemoveCasesFew(TransformerMixin):
	'''this class remove the cases that are too few, less than the threshold in the 
	previous function:##drop cases where have <=5 records of each credId'''
	def transform(self, X, **transform_params):
		return X[X.loc[:,'case_thres'] == 1]	
	#the class first does the fit, and then transform based on the fit function
	def fit(self, X, y=None, **fit_params):
		return self

class DropVarsFew(TransformerMixin):
	'''this class drop cols/variables that having too few cases::##drop some columns with too few records, less than 20
	'''
	def transform(self, X, **transform_params):
		return X.drop(['case_thres','body_bodyFat','food_water'],axis=1)

	def fit(self, X, y=None, **fit_params):
		return self	

class FillAllCategoricalNoneAsNA(TransformerMixin):
	def fit(self, X, y=None, **fit_params):
		return self
	def transform(self, X, **transform_params):
		categorical_vars = ['mobiles', 'trackers','platforms']
		for var in categorical_vars:
			X.loc[:, var] = X.loc[:,var].apply(lambda x: 'NA' if x is None else x)		
		return X

class FixTypo(TransformerMixin):
	def fit(self, X,y=None, **fit_params):
		return self
	def transform(self, X, **transform_params):
		#this way, the X will be changed, and be careful about this: the best way is to return the direct array and append it to the dataframe via featureUnion
		#donot change directly on the X-dataframe or array will be better
		X.loc[:,'trackers'] = X['trackers'].apply(lambda x: u'misfit' if x == u'mistfit' else x)
		return X


class GetDummies(TransformerMixin):
	def fit(self, X, y=None, **fit_params):
		return self
	def transform(self, X, **transform_params):
		X = pd.concat([X, pd.get_dummies(X['mobiles'], prefix='mobiles')], axis = 1)
		X = pd.concat([X, pd.get_dummies(X['trackers'], prefix='trackers')], axis = 1)
		X = pd.concat([X, pd.get_dummies(X['platforms'], prefix='platforms')], axis = 1)
		return X

#first pipleline
filter_few=make_pipeline(RemoveCasesFew(),DropVarsFew(),FixTypo(),\
 FillAllCategoricalNoneAsNA(), GetDummies())
# filter_few=make_pipeline(RemoveCasesFew(),DropVarsFew(),FixTypo(),\
#  FillAllCategoricalNoneAsNA())
# filter_few_no_dummy=make_pipeline(RemoveCasesFew(),DropVarsFew(),FixTypo(),\
#  FillAllCategoricalNoneAsNA())

fdf=filter_few.fit_transform(fdf)
print 'fdf shape now is:', fdf.shape


# ##########################model training and selection##########################################
from sklearn.ensemble import RandomForestClassifier

#segment data
fdf['data_type'] = fdf['gender'].apply(lambda x: 2 if np.isnan(x) else 0)
oob = fdf[fdf['data_type'] == 2] #2 in data_type means oob
inb = fdf[fdf['data_type'] == 0]
# print inb.shape
# print oob.shape
#separate the inb data into train and test
#0 is the test, and 1 means the train
np.random.seed(197)
inb.loc[:, 'data_type'] = np.array(pd.Series((np.random.uniform(0, 1, len(inb)) <= .85)).apply(lambda x: 1 if x==True else 0))
#separate into test and train
train = inb[inb['data_type'] == 1]
test = inb[inb['data_type'] == 0]

#get y value
y_train = train.loc[:,'gender']
#test here is for estimating the oob
y_test = test.loc[:,'gender']
y_oob = oob.loc[:,'gender']


class RemoveCasesHaveLessThanThresCols(TransformerMixin):
	'''remove any cases that have <= thres columns of each Row/credId when all other columns are missing'''
	def __init__(self, thres):
		self.thres = thres
	def fit(self, X, y=None,**fit_params):
		return self
	def transform(self, X, **transform_params):
		return X[X.isnull().sum(axis=1)>self.thres]
# ---right now, NOT remove these cases above


class FillNAsForAllFloatsVars(BaseEstimator,TransformerMixin):
	'''filling all the NAs now based on train by mean for all the floats'''
	def __init__(self):
		self.dct={}

	def fit(self, X, y=None, **fit_params):
		self.dct=dict(X.mean())
		#remove all the categorical variables that cannot use the Mean
		#donot need to add trackers_ and mobiles_, since NAs/Nones of them have been filled out
		#and there are no missing values for them, even they are float/int now
		categorical_vars = ['gender','data_type']
		for key in categorical_vars:
			self.dct.pop(key)
		return self

	def transform(self, X, **transform_params):
		return X.fillna(self.dct)		

class DropUnusedVars(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None, **fit_params):
		return self
	def transform(self, X, **transform_params):
		#drop trackers_NA, mobiles_NA to eliminate colinearity
		return X.drop(['data_type', 'gender', 'platforms', 'mobiles', 'trackers', 'trackers_NA', 'mobiles_NA','platforms_NA'], axis = 1)
		#do NOT drop trackers_NA, mobiles_NA
		# return X.drop(['data_type', 'gender', 'platforms', 'mobiles', 'trackers'], axis = 1)

#not including RemoveCasesHaveLessThanThresCols Yet in the following pipeline
#random forest classifier
#donot need pca etc., for Rf, the important parameter values: n_estimators, and max_features
#you can also vary min_samples_split a little bit
fill_na_floats_drop_unused = make_pipeline(FillNAsForAllFloatsVars(), DropUnusedVars())
fill_na_floats_drop_unused.fit(train)
train = fill_na_floats_drop_unused.transform(train)
test = fill_na_floats_drop_unused.transform(test)
oob = fill_na_floats_drop_unused.transform(oob)


# ###random forest classifier
# rfc = make_pipeline(RandomForestClassifier(oob_score=True))
# #rfc need train and y, and would use y in the last step to fit the RF model
# # rfc.fit(train, y_train)
# #parameter grids
# n_estimators = [20, 50, 100, 200]
# max_features = [2,3,5,6,10,15,20]
# min_samples_split = [1,2,5]

# estimator = GridSearchCV(rfc, param_grid = dict(randomforestclassifier__n_estimators = n_estimators, \
# 	randomforestclassifier__max_features=max_features, randomforestclassifier__min_samples_split=min_samples_split), cv = 10)
# estimator.fit(train, y_train)

# y_train_predict = estimator.predict(train)
# y_test_predict = estimator.predict(test)

# print 'the best parameter setting is:', estimator.best_estimator_
# print 'the best CV score in the GridSearchCV is:', estimator.best_score_
# #best CV score is: 0.868035190616
# print 'the in-bag prediction error rate is:', (y_train_predict == y_train).sum()/float(y_train.shape[0])
# #0.988269794721
# print 'the out-bag prediction error rate is:', (y_test_predict == y_test).sum()/float(y_test.shape[0])
# #0.866666666667


# ###import SVM predictor--too slow to fit
# from sklearn.svm import SVC
# #parameter to change: 
# C=[0.1, 0.3, 1, 3, 9]; kernel = ['linear', 'rbf', 'sigmoid']; coef0=[0, 0.03, 0.1, 0.3]

# svm = GridSearchCV(SVC(), param_grid = dict(C = C, \
# 	kernel=kernel, coef0=coef0), cv = 10)
# svm.fit(train, y_train)

# y_train_predict = svm.predict(train)
# y_test_predict = svm.predict(test)

# print 'the best parameter setting is:', svm.best_estimator_
# print 'the best CV score in the GridSearchCV is:', svm.best_score_
# #best CV score is: 
# print 'the in-bag prediction error rate is:', (y_train_predict == y_train).sum()/float(y_train.shape[0])
# #
# print 'the out-bag prediction error rate is:', (y_test_predict == y_test).sum()/float(y_test.shape[0])
# #


###import logistic regression classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler as Scale

#parameter to change: 
penalty = ['l1','l2']; C=[0.1, 0.3, 1, 3, 9]; intercept_scaling = [0.1,0.3,1,3,9]
#scaling or not
FLOAT_VARS = [u'activity_activeMinutes', u'activity_calories', u'activity_distance', u'activity_nonactiveMinutes', u'activity_steps', u'food_calories', u'food_carbs', u'food_fat', u'food_fiber', u'food_protein', \
		u'food_sodium', u'sleep_asleep', u'sleep_awake', u'sleep_duration', u'height', u'weight', u'age', u'bmi']

class ScalingFloat(BaseEstimator, TransformerMixin):
	'''you can do the following, or make a pipeline with select float and scaler directly'''
	def __init__(self):
		self.scaler = Scale(with_mean=True, with_std=True, copy=True)
	def fit(self, X, y=None):
		float_df = X.loc[:,FLOAT_VARS]
		self.scaler.fit(float_df)
		return self
	def transform(self, X, y=None):
		float_df = X.loc[:,FLOAT_VARS]
		return self.scaler.transform(float_df)

class SelectCategoryVars(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):
		res=[]
		for item in X.columns:
			if item not in FLOAT_VARS: res.append(item)
		return np.array(X.loc[:,res])

combine_feature = FeatureUnion([("scalingfloat", ScalingFloat()), ("selectcategory", SelectCategoryVars())])
combine_feature.fit(train)
train = combine_feature.transform(train)
test = combine_feature.transform(test)


lrc = GridSearchCV(LogisticRegression(), param_grid = dict(C = C, \
	penalty=penalty, intercept_scaling=intercept_scaling), cv = 10)
lrc.fit(train, y_train)

y_train_predict = lrc.predict(train)
y_test_predict = lrc.predict(test)

#scaled results: it seems that though RF has higher accuracy, but logistic regression has no overfitting, which can be a huge advantage
# print 'the best parameter setting is:', lrc.best_estimator_
# # the best parameter setting is: LogisticRegression(C=9, class_weight=None, dual=False, fit_intercept=True,
#           # intercept_scaling=3, penalty='l1', random_state=None, tol=0.0001)
# print 'the best CV score in the GridSearchCV is:', lrc.best_score_
# #best CV score is: 0.800586510264
# print 'the in-bag prediction error rate is:', (y_train_predict == y_train).sum()/float(y_train.shape[0])
# #0.850439882698
# print 'the out-bag prediction error rate is:', (y_test_predict == y_test).sum()/float(y_test.shape[0])
# #0.85


# #unscaled results
# print 'the best parameter setting is:', lrc.best_estimator_
# # the best parameter setting is: LogisticRegression(C=9, class_weight=None, dual=False, fit_intercept=True,
# #           intercept_scaling=9, penalty='l1', random_state=None, tol=0.0001)
# print 'the best CV score in the GridSearchCV is:', lrc.best_score_
# #best CV score is: 0.797653958944
# print 'the in-bag prediction error rate is:', (y_train_predict == y_train).sum()/float(y_train.shape[0])
# #0.853372434018
# print 'the out-bag prediction error rate is:', (y_test_predict == y_test).sum()/float(y_test.shape[0])
# #0.85


# # ###make predictoins for cases missing gender values###
# # y_oob_predict = estimator.predict(oob)

