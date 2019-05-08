from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import manifold


# Reading the xlsx file using pandas read_excel, checking the null values and impute those instances
cancer_data = pd.read_excel('breast-cancer-wisconsin.xlsx', sheet_name="Sheet1")
#interpolation
cancer_data = cancer_data.interpolate(method ='linear', limit_direction ='forward')

cancer_data['class'].replace([2, 4], [0,1], inplace=True)
cancer_data.head()


'Scaling'
features = ['thickness', 'uniCelS', 'uniCelShape', 'marAdh', 'epiCelSize', 'bareNuc', 'blaChroma', 'normNuc', 'mitoses']
# Separating out the features
x = cancer_data.loc[:, features].values
# Separating out the target
y = cancer_data.loc[:,['class']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

# Task 1(a)
# Random Initialization
tsne = TSNE(verbose=1, perplexity=20, n_iter= 4000,init='random')
tsne = tsne.fit_transform(x)
plt.scatter(tsne[:,0],tsne[:,1],  c = cancer_data['class'], cmap = "winter", edgecolor = "None", alpha=0.35)
plt.title('t-SNE Scatter Plot')

# PCA Initialization
tsne = TSNE(verbose=1, perplexity=20, n_iter= 4000,init='pca')
tsne = tsne.fit_transform(x)
plt.scatter(tsne[:,0],tsne[:,1],  c = pcaDf['class'], cmap = "winter", edgecolor = "None", alpha=0.35)
plt.title('t-SNE Scatter Plot')

#Task  ---------need to write----------


# Task 1(b)
# Reading the xlsx file using pandas read_excel, checking the null values and impute those instances
mice_data = pd.read_excel('Data_Cortex_Nuclear.xls', sheet_name="Hoja1")
#mice_data.isnull().sum()

#interpolation
mice_data = mice_data.interpolate(method ='linear', limit_direction ='forward')
median = mice_data['BCL2_N'].median()
mice_data.fillna(median, inplace=True)
#mice_data.isnull().sum()
mice_data.head()

sub_grp_cSCs = mice_data[mice_data['class'] == 'c-SC-s']
#sub_grp_cSCs.head()
len(sub_grp_cSCs.index)


sub_grp_tSCs = mice_data[mice_data['class'] == 't-SC-s']
#sub_grp_cSCs.head()
len(sub_grp_tSCs.index)

sbgrp_list = ['c-SC-s','t-SC-s']
sub_grp_cSCs_tSCs = mice_data[mice_data['class'].isin(sbgrp_list)]
sub_grp_cSCs_tSCs.head()

# Separating out the features
x = sub_grp_cSCs_tSCs.iloc[:,1 :-4].values
#x.head()
y = sub_grp_cSCs_tSCs.loc[:,['class']].values


pca = PCA(n_components=2).fit(x)
principalComponents = pca.transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf.head()


principalDf.reset_index(drop=True, inplace=True)
sub_grp_cSCs_tSCs.reset_index(drop=True, inplace=True)
finalDf = pd.concat([principalDf, sub_grp_cSCs_tSCs['class']], axis = 1)
finalDf.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['c-SC-s', 't-SC-s']
colors = ['g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

iso = manifold.Isomap(n_components=2, n_neighbors=6)
iso.fit(x)
manifold_2Da = iso.transform(x)
manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])

# Left with 2 dimensions
manifold_2D.head()

finalDfIso = pd.concat([manifold_2D, sub_grp_cSCs_tSCs['class']], axis = 1)
finalDfIso.head()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['c-SC-s', 't-SC-s']
colors = ['g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDfIso['class'] == target
    ax.scatter(finalDfIso.loc[indicesToKeep, 'Component 1']
               , finalDfIso.loc[indicesToKeep, 'Component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

sub_grp_cSCs_tSCs['class'].replace(['c-SC-s', 't-SC-s'], [0,1], inplace=True)
tsne = TSNE(verbose=1, perplexity=20, n_iter= 4000)
tsne = tsne.fit_transform(x)
plt.scatter(tsne[:,0],tsne[:,1],  c = sub_grp_cSCs_tSCs['class'], cmap = "winter", edgecolor = "None", alpha=0.35)
plt.title('t-SNE Scatter Plot')


