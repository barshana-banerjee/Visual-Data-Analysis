import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import networkx as nx

# Reading the xlsx file using pandas read_excel, checking the null values and impute those instances
cancer_data = pd.read_excel('breast-cancer-wisconsin.xlsx', sheet_name="Sheet1")
#interpolation
cancer_data = cancer_data.interpolate(method ='linear', limit_direction ='forward')

cancer_data.head()

correlations = cancer_data.corr('pearson')
print(type(correlations))
# Task 4(a) - plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(cancer_data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(cancer_data.columns)
ax.set_yticklabels(cancer_data.columns)
plt.show()

plt.figure(figsize=(20,20))

links = correlations.stack().reset_index()
links.columns = ['var1', 'var2','value']
 
# Keep only correlation over a threshold
links_filtered=links.loc[ (links['value'] > 0.6)  ]
 
# Task 4(b) - Build your graph
G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
 
# Plot the network:
nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)

plt.savefig('graph.png')

edge_list = []
for edge in G.edges:
    edge_list.append(edge)

#Task 4(d) - A circular layout
plt.figure(figsize=(20,20))
nx.circular_layout(G)
color_map = []
for val in links['value']:
    if abs(val) > 0.9:
        color_map.append('blue')
    elif (abs(val) > 0.8 and abs(val) <= 0.9):
        color_map.append('red')
    elif (abs(val) > 0.6 and abs(val) <= 0.8):
        color_map.append('yellow')
    else: 
        color_map.append('green')    
nx.draw(G,node_color = color_map,with_labels = True)
plt.savefig('circular.png')
#plt.savefig("cirgraph.png", dpi=1000)

#Task 4(e) - code and mitoes are disconnected. It shows that they are not correlated to any other node.
# i will propose thickness, unicels and uniCelShape to predict the class




