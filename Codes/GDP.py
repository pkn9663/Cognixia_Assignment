"""

*------------************************************************************--------------------------*
*                                                                                                  *
*   1. Add columns with standardized variables which have a mean of 0 and a variance of 1          *
*   2. Plot the elbow curve for these normalised variables to determine how many clusters to use   *
*   3. Create k-means clusters                                                                     *
*   4. Plot the k-means clusters                                                                   *
*   5. Calculate average GDP, average co2 and number of countries by cluster                       *
*   6. Plot the dendrogram chart for these normalised var to determine how many clusters to use    *
*-----------************************************************************---------------------------*

"""
# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel("C:\\Users\\prave\\OneDrive\\Documents\\GitHub\\Assignment\\Datasets\\Cluster_Problem_Dataset.xlsx")
data.head()

data.info()
data.dropna(inplace= True)

data.drop(["CountryName"] , axis= 1 , inplace= True)
data.info()

# feature Scaling
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = pd.DataFrame(data)
X = scalar.fit_transform(X)
X = pd.DataFrame(X , columns= ["co2" , "gdp"])
# concate DataFrames
data = pd.concat([data , X] , axis = 1 , ignore_index= False)
data.head(5)

# Elbow plot
wcss = []
from sklearn.cluster import KMeans
for i in range(1 , 11):
    km = KMeans(n_clusters = i , init= "k-means++" , n_init= 10 , max_iter= 300 , random_state= 0 , n_jobs = -1)
    km.fit(data.iloc[: , [2,3]].values)
    wcss.append(km.inertia_)
plt.figure(figsize=(8 , 5) , dpi = 100 , facecolor= "yellow" , edgecolor= "blue")
plt.plot(range(1 , 11) , wcss , color = "brown" , marker = "o" , markerfacecolor = "black" ,markersize = 10)
plt.show() # cluster = 3

# Dendrogram
from scipy.cluster import hierarchy
dandrons = hierarchy.linkage(data.iloc[: , [2,3]].values , method = "ward")
hierarchy.dendrogram(dandrons)
plt.show() # cluster = 3

""" KMeans Observations """
km = KMeans(n_clusters= 3 , init= "k-means++" , n_init = 10 , max_iter= 300 , n_jobs=-1 , random_state= 0)
km.fit(data.iloc[: , [2,3]].values)
y_pred = km.fit_predict(data.iloc[: , [2,3]].values)

data["cluster"] = y_pred
g = data.groupby("cluster")

k0 = g.get_group(0)
k1 = g.get_group(1)
k2 = g.get_group(2)
km.cluster_centers_
# K-Means Visualisations
plt.figure(figsize=(8 , 5) , dpi = 150 , facecolor= "yellow")
plt.title("K-Means Observations")
plt.scatter(k0.values[: , 2] , k0.values[: , 3] , c = "blue" , marker = "o" , label = "KMeans-1")
plt.scatter(k1.values[: , 2] , k1.values[: , 3] , c = "brown" , marker = "o" , label = "KMeans-2")
plt.scatter(k2.values[: , 2] , k2.values[: , 3] , c = "green" , marker = "o" , label = "KMeans-3")
plt.scatter(km.cluster_centers_[: , 0] , km.cluster_centers_[: , 1] , s = 100 ,c = "black" , marker = "*" , label = "Centroid")
plt.xlabel("CO2 data")
plt.ylabel("GDP data")
plt.legend()
plt.show()

""" Heirarchical observations """

from sklearn.cluster import AgglomerativeClustering as ACS
cls = ACS(n_clusters=3 , affinity= "euclidean" , linkage= "ward")
cls.fit(data.values[: , [2,3]] )

y_pred = cls.fit_predict(data.values[: , [2,3]])
data["cluster"] = y_pred

g = data.groupby("cluster")

h0 = g.get_group(0)
h1 = g.get_group(1)
h2 = g.get_group(2)

# Hierarchical Visualisations
plt.figure(figsize=(8 , 5) , dpi = 150 , facecolor= "yellow")
plt.title("Hierarchical Observations")
plt.scatter(h0.values[: , 2] , h0.values[: , 3] , c = "blue" , marker = "o" , label = "HClust-1")
plt.scatter(h1.values[: , 2] , h1.values[: , 3] , c = "brown" , marker = "o" , label = "HClust-2")
plt.scatter(h2.values[: , 2] , h2.values[: , 3] , c = "green" , marker = "o" , label = "HClust-3")
plt.xlabel("CO2 data")
plt.ylabel("GDP data")
plt.legend()
plt.show()

""" Finding the average of CO2 & GDP values """
print(f"Average of CO2 grouped by clusters:{np.average(h0.values[: , 0])} ,{np.average(h1.values[: , 0])} , {np.average(h2.values[: , 0])}")
print(f"Average of GDP grouped by clusters:{np.average(h0.values[: , 1])} ,{np.average(h1.values[: , 1])} , {np.average(h2.values[: , 1])}")
