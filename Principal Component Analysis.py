''' 
Submitted By: 
	Hemanth Kumar Reddy Mayaluru
	Barshana Banerjee
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Exercise 1(a) - Reading the xlsx file using pandas read_excel, checking the null values and impute those instances
cancer_data = pd.read_excel('breast-cancer-wisconsin.xlsx', sheet_name="Sheet1")
cancer_data.head()

# Number of null values in each column
cancer_data.isnull().sum()

# We can see there are only 16 null values and that too in tonly one column 'bareNuc' . Here we used median value of the column 'bareNuc' for imputation, because mot of the values are 1 in this column and the values which are null has 2 as the 'class' for which 'bareNuc' value is 1 in the other cases with calss 2
median = cancer_data['bareNuc'].median()
cancer_data.fillna(median, inplace=True)
cancer_data.isnull().sum()

cancer_data['class'].replace([2, 4], ['benign', 'malignant'], inplace=True)
cancer_data.head()

'Scaling'
features = ['thickness', 'uniCelS', 'uniCelShape', 'marAdh', 'epiCelSize', 'bareNuc', 'blaChroma', 'normNuc', 'mitoses']
# Separating out the features
x = cancer_data.loc[:, features].values
#print(x)
# Separating out the target
y = cancer_data.loc[:,['class']].values
#print(y)
# Standardizing the features
x = StandardScaler().fit_transform(x)
#print('standard x: ', x)


#1(b)
pca = PCA(n_components=5).fit(x)
principalComponents = pca.transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])


principalDf.head()

finalDf = pd.concat([principalDf, cancer_data[['class']]], axis = 1)
#finalDf.to_csv('pca_sheet.csv')
finalDf.head()
#print(finalDf.loc[[65]])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['benign', 'malignant']
colors = ['g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');



#Task 1(c)
sns.pairplot(finalDf, palette=('b','g'),hue="class")



# 1(d)
feat = pd.DataFrame(pca.components_,columns=features,index = ['PC-1','PC-2','PC-3','PC-4','PC-5'])
feat.head()




'''
1(d) - We can see from the plot above that the Principal Component mode PC-1 shows strongest difference between benign and 
malignant samples

The original variables that have the highest and lowest weight numbers is determined by the values of larger magnitude, 
the farthest from zero in either direction from the above table

For PC-1:
Highest Weight: uniCelS (0.381190)
Lowest Weight: mitoses (0.229818)

For PC-2:
Highest Weight: mitoses (0.908394)
Lowest Weight: marAdh (-0.044719)

For PC-3:
Highest Weight: thickness (-0.862217)
Lowest Weight: bareNuc (-0.003671)

For PC-4:
Highest Weight: bareNuc (-0.543177)
Lowest Weight: blaChroma (-0.006990)

For PC-5:
Highest Weight: epiCelSize (0.682582)
Lowest Weight: thickness (-0.071928)
'''



'''
1(e)
From the scatter plot found that the data row 689 is an outlier (pc-1 vs pc-2).
'''
finalDf_drop = finalDf.drop(finalDf.index[[689]])
#finalDf_drop.to_csv('new_dropped.csv')
sns.pairplot(finalDf_drop, palette=('b','g'),hue="class")



'''
1(f):
If the variables of a dataset have very different ranges, PCA will load on the large variances. 

In our example, one variable x1: [1000; 2000] and another one x2: [1; 5], x1 have more impact and x2 will have minimal 
impact during the principal component generation. Only x1 explains all the variance in the data. The principal components 
are linear combinations of the original variables of the data table analyzed. If the relationships between the variables 
analyzed are not linear, the values of correlation coefficients can be  lower. 
Thus, it makes sense to sometimes rescale the original variables to "linearize" these relationships.

Log-transform or z-score are 2 such methods to scale the data
'''



'''
1(g) - 
PCA tends to outperform LDA if the number of samples per class is relatively small. In our case we have less number of samples.
hence LDA is not useful.
'''
X_ = cancer_data[cancer_data.columns[:-1]]
y_ = cancer_data['class']
lda = LDA(n_components=1)  
lda1 = lda.fit_transform(X_, y_) 
princ = finalDf['principal component 1']
princ = princ.as_matrix(columns=None)
sns.scatterplot(x=lda1.tolist(), y=princ.tolist())

