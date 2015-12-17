import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re

train = pd.read_csv('train.csv', header = 0)

#name is a series
name = train['Name']
# number of names in the training data
num_names = name.nunique()
#extract the title information from the name
title = name.str.findall('M.{0,6}\.')

print title.apply(str).nunique()
#the result is 10
print title.apply(str).unique()
# array(["['Mr.']", "['Mrs.']", "['Miss.']", "['Master.']", '[]',
#        "['Meo, Mr.']", "['Mme.']", "['Ms.']", "['Major.']", "['Mlle.']"], dtype=object)
#so has to treat each case differently
#check the frequency of each title in the data
title_str = title.apply(str)
title_str.value_counts()
# ['Mr.']         516
# ['Miss.']       182
# ['Mrs.']        125
# ['Master.']      40
# []               21
# ['Major.']        2
# ['Mlle.']         2
# ['Ms.']           1
# ['Meo, Mr.']      1
# ['Mme.']          1
# Mr. Mrs. Miss Master. should be kept, others should be transfered to these four
# Master is replaced by Mister(Mr.) in 19th century, titanic happned in 1912, so master. meaning boys and young men
#major -> Mr.
#Mlle. -> Miss.
#Ms    -> Mrs.
# Meo, Mr.  --this is wrong separation -> Mr.
# Mme. Madame-> Mrs.

title_str = title_str.apply(lambda x: x[2:-2])
# Note that now title_str has ['Mr.'], all strings inside are strings, including \' and \'
#major -> Mr.
print title_str[title_str == 'Major.']
print title_str[title_str == 'Major.'].index
index_major = title_str[title_str == 'Major.'].index
title_str.loc[index_major] = 'Mr.'
#Mlle. -> Miss.
print title_str[title_str == 'Mlle.']
print title_str[title_str == 'Mlle.'].index
index_Mlle = title_str[title_str == 'Mlle.'].index
title_str.loc[index_Mlle] = 'Miss.'
#Ms    -> Mrs.
print title_str[title_str == 'Ms.']
print title_str[title_str == 'Ms.'].index
index_Ms = title_str[title_str == 'Ms.'].index
title_str.loc[index_Ms] = 'Mrs.'
#Meo, Mr.  --this is wrong separation -> Mr.
print title_str[title_str == 'Meo, Mr.']
print title_str[title_str == 'Meo, Mr.'].index
index_Meo = title_str[title_str == 'Meo, Mr.'].index
title_str.loc[index_Meo] = 'Mr.'
# Mme.(Madame)-> Mrs.
print title_str[title_str == 'Mme.']
print title_str[title_str == 'Mme.'].index
index_Mme = title_str[title_str == 'Mme.'].index
title_str.loc[index_Mme] = 'Mrs.'


# now working on the empty list
empty_index = title_str[title_str == ''].index
name_fix = name.loc[empty_index]
print name_fix

# Rev.-> Mr.
# Dr. -> Mr.





