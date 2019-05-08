#!/usr/bin/env python
# coding: utf-8

# In[47]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

excel_file = 'chronic_kidney_disease_numerical.xls'
data = pd.read_excel(excel_file)

d = pd.melt(data, id_vars='class',value_vars=None,var_name=None, value_name='value', col_level=None)
print(d)

b = sns.boxplot(data = df_melt,
                hue = 'class', # different colors for different 'cls'
                x = 'columns',
                y = 'value',
                order = ['age', # custom order of boxplots
                         'blood',
                         'albumin',
                         'sugar',
                         'blood pressure',
                        'blood glucose random',
                        'blood urea',
                        'white blood cell count',
                        'red blood cell count'
                        ])

plt.title('Boxplot grouped by class') # You can change the title here
b.set_xticklabels(b.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[ ]:




