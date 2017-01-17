#This is a script for taking a pandas dataframe of edges and making a network plot using networkx and pygraphviz.
##To run this code you will need you will need networkx (pip install), graphviz (brew or yum install), and pygraphviz (pip install)

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot, view_pygraphviz

#bring in a dataframe with the columns (Toy dataset): 
##From (starting position), To (ending position), Freq (frequency of occurence in data)
A=pd.DataFrame(data={'From':[1,2,3,4,5],'To':[5,4,1,1,1], 'Freq':[1,1,2,1,3]})


#Get values into list format so they can be added as edges
edges=[]
for index, row in A.iterrows():
    x=[]
    x.append(row['From'])
    x.append(row['To'])
    x.append(row['Freq'])
    edges.append(x)

#open a directional graph
G = nx.DiGraph()

#Add the edges and weights of the arrows (taken from Freq)
for (u, v, w) in edges:
    G.add_edge(u,v,penwidth=w)

#Draw and save the graph, it will automatically open as well
##prog can be different graphviz display layouts: dot, neato, twopi, circo, fdp, or sfpd
## examples of what this toy dataset looks like with different layouts can be seen in data_viz/netToy_displays 
outFile = open('fileName.pdf', 'wb')
view_pygraphviz(G,prog='circo',path=outFile)




