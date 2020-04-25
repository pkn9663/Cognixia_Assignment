""" ----------- K-Means ---------------- """
"""
1. Add columns with standardized variables which have a mean of 0 and a variance of 1
2. Plot the elbow curve for these normalised variables to determine how many clusters to use
3. Create k-means clusters
4. Plot the k-means clusters
5. Calculate average GDP, average co2 and number of countries by cluster
"""

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel("C:/Users/prave/OneDrive/Documents/GitHub/Assignment/Datasets/Cluster_Problem_Dataset.xlsx")
data.head()

data.describe()
data.info()

# replacing the NaN value
data.dropna(inplace= True)
X = data.iloc[: , :2].values
X

# feature scalling
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(X)
X
X = pd.DataFrame(X , columns = ["co2" , "GDP"])

# adding Standard scalar column to DataFrame
data = pd.concat([data , X] ,axis=1 , ignore_index= True)
data.head()
data.info()

# fetching few columns from DataFrame
df = data.iloc[: , [3,4]].values
df

# Elbow method to know the Cluster value
from sklearn.cluster import KMeans
wssc = []
for i in range(1 , 11):
    km = KMeans(n_clusters= i , init= "k-means++" , n_init = 10 , max_iter= 300 , random_state= 0)
    km.fit(df)
    wssc.append(km.inertia_)
plt.figure(figsize=(5 , 8) , facecolor= "yellow" , edgecolor= "blue")
plt.plot(range(1 , 11) , wssc , c = "red" , marker = "o" , ls = "--")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.show() # from plot n_cluster = 3

# K-Means Cluster
km = KMeans(n_clusters= 3 , init = "k-means++" , n_init= 10 , max_iter= 300 , random_state= 0)
y_pred = km.fit_predict(df)
y_pred
data["cluster"] = y_pred

# grouping as per the cluster group
g = data.groupby("cluster")
c0 = g.get_group(0)
c1 = g.get_group(1)
c2 = g.get_group(2)
km.cluster_centers_

# visualising

plt.figure(figsize=(10 , 8) , dpi = 50 , facecolor= "yellow" , edgecolor= "blue")
plt.title("K-Means observations")
plt.scatter(c0.iloc[: , 3].values , c0.iloc[: , 4].values ,s = 100, c = "blue" , label = "K-Means 1")
plt.scatter(c1.iloc[: , 3].values , c1.iloc[: , 4].values , s = 100,c = "green" , label = "K-Means 2")
plt.scatter(c2.iloc[: , 3].values , c2.iloc[: , 4].values , s = 100,c = "red" , label = "K-Means 3")
plt.scatter(km.cluster_centers_[: , 0] , km.cluster_centers_[: , 1] , s = 100 , marker= "*" , c = "black" , label = "centroid")
plt.xlabel("CO2 observations")
plt.ylabel("GDP observations")
plt.legend()
plt.show()


"""----------------- Hierarchical Clustering -----------------------"""

"""
1. Add columns with standardized variables which have a mean of 0 and a variance of 1
2. Plot the dedrogram chart for these normalised variables to determine how many clusters to use
3. Create hierarchical clusters
4. Plot the Hierarchical clusters
5. Calculate average GDP, average co2 and number of countries by cluster
"""

# Dendrogram
from scipy.cluster import hierarchy
dendro = hierarchy.linkage(df)
hierarchy.dendrogram(dendro)
plt.show() # as per observation cluster = 2

# hierarchical clusters
from sklearn.cluster import AgglomerativeClustering
cls = AgglomerativeClustering(n_clusters=2 , affinity= "euclidean" , linkage= "ward")
y_pred = cls.fit_predict(df)

data["cluster"] = y_pred

# grouping as per the cluster group
g = data.groupby("cluster")
c0 = g.get_group(0)
c1 = g.get_group(1)

# visualising

plt.figure(figsize=(10 , 8) , dpi = 50 , facecolor= "yellow" , edgecolor= "blue")
plt.title("Hierarchical Clusters Observations")
plt.scatter(c0.iloc[: , 3].values , c0.iloc[: , 4].values ,s = 100, c = "blue" , label = "K-Means 1")
plt.scatter(c1.iloc[: , 3].values , c1.iloc[: , 4].values , s = 100,c = "green" , label = "K-Means 2")
plt.xlabel("CO2 observations")
plt.ylabel("GDP observations")
plt.legend()
plt.show()


