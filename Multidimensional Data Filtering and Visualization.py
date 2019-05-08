

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing

# 1(a) - Reading the data and printing first few rows
data = pd.read_csv('winequality-red-actual.csv')
data.head()


# 1(b) - Histogram for the distribution of column "quality" and it's range
sns.distplot(data['quality'],  color='green');
range_q = data["quality"].max() - data["quality"].min()
print('Range: ', range_q)


#1(c) - classification of quality into "low", "medium", and "high"
#bins = (0,data['quality'].nsmallest(2).min(),data['quality'].nlargest(2).min()-1,data['quality'].nlargest(2).min())
bins = (0,4,7,8)
data.insert(12,'quality bin',1)
group_names = ['low', 'medium', 'high']
data['quality bin'] = pd.cut(data['quality'], bins = bins, labels = group_names)
data.drop('quality',1)
del data['quality']
data.head()
data.to_csv('winequality-red-actual.csv',index=False)


#1(d) - filtered data frame in which the medium-quality wines are omitted
data_filtered = data.loc[data['quality bin'] != 'medium']
#data_filtered.head(20)
data_filtered2 = data_filtered
data_filtered3 = data_filtered
data_filtered.head(20)


#1(e) - scatterplot matrix
ax = sns.pairplot(hue='quality bin', markers=False, data=data_filtered)


''' 1(f) - Based on the visualization, below are the five attributes that appear to best distinguish between high and
           low quality:
                - fixed acidity
                - volatile acidity
                - pH
                - sulphates
                - alcohol           
'''

# 1(g) -  automated feature selection technique to identify five attributes that distinguish between high and low quality

#Create features and target
X = data_filtered2
y = 'quality bin'

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=5)
data_filtered2['quality bin'] = np.where(data_filtered2['quality bin'] == 'high', 1, 0)

# Apply the SelectKBest object to the features and target
X_kbest = fvalue_selector.fit_transform(data_filtered2,data_filtered2['quality bin'])
cols=fvalue_selector.get_support(indices=True)
colname=[]
print("Features:")
for i in cols:
    colname.append(data_filtered2.columns[i-1])  
    print(" - ", data_filtered2.columns[i-1])

# Show results
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])


#1(h) - Create a matrix similar to the one in Fig. 1. Legend (1  - for high, 0 - for low)
g = sns.PairGrid(data_filtered, hue='quality bin',y_vars=['volatile acidity','citric acid','pH','sulphates','alcohol'],x_vars =['volatile acidity','citric acid','pH','sulphates','alcohol'])
g.map_upper(plt.scatter)
g.map_lower(sns.regplot,scatter=False)
g.map_diag(sns.kdeplot, lw=3, legend=True);
g.add_legend()



