import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#K-menas on a randomly generated dataset
np.random.seed(0)

X, y = make_blobs(n_samples = 5000, centers = [[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
#n_sample = The total number points equally divided among clusters. Value: 5000
#centers = The number of centers to generate, or fixed center locations. Value: [4,4], [-2, -1], [2, -3], [1, 1]
#cluster_std = The standard deviation of the clusters. Value: 0.9

plt.scatter(X[:, 0], X[:, 1], marker = ".")

#Setting up K-means
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
#İnitialize KMeans, where the output parameter is called k_means.

#Fit model
k_means.fit(X)

#Grap Labels
k_means_labels = k_means.labels_
k_means_labels

#Get coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#Plot model
fig = plt.figure(figsize = (6,4))

#We use set(k_means_labels) to get unique labels
colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_labels))))

#Create a plot
ax = fig.add_subplot(1,1,1)

#k range 0-3, match the possible clusters that eachs data points

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])),colors):
    #Create list of all data point, where the data points that are in the cluster.
    #Labeled as True, else they are labeled as false.
    my_members = (k_means_labels == k)

    #Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    #Plots the datapoints
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor = col, marker = ".")

    #Plots the centroids with specified color with darker outline
    ax.plot(cluster_center[0],cluster_center[1],"o",markerfacecolor = col, markersize = 6)
#title
ax.set_title("KMeans")
#Remove x-axis ticks
ax.set_xticks(())
#Remove y-axis ticks
ax.set_yticks(())
plt.show()

#Cluster dataset into 3 Cluster.
k_means_3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
#Fit
k_means_3.fit(X)
#labels
k_means_3_labels = k_means_3.labels_
k_means_3_labels
#cluster center
k_means_3_cluster_centers = k_means_3.cluster_centers_
k_means_3_cluster_centers

#Visual Plot
fig = plt.figure(figsize = (6,4))
colors = plt.cm.Spectral(np.linspace(0,1,len(set(k_means_3_labels))))

#Create plot
ax = fig.add_subplot(1,1,1)

for k, col in zip(range(len(k_means_3.cluster_centers_)), colors):
    my_members = (k_means_3_labels == k)
    cluster_center = k_means_3_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
ax.set_title("KMeans_3")
ax.set_xticks(())
ax.set_yticks(())
plt.show()

#Load data
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()
#You can download data via this link
#https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv

#Pre-processing
#Address categorical
df = cust_df.drop("Address", axis = 1)
df.head()

#Normalizing over the Standad Deviation
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)

Clus_dataset = StandardScaler().fit_transform(X)
Clus_dataset

#Modeling
#apply k- means on our dataset
clusterNum = 3
k_means = KMeans(init = "k-means++",n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

#Insights
df["Clus_km"] = labels
df.head(10)

#Check Centroid Values
df.groupby("Clus_km").mean()

#Distribution of custormers base on their age, income

#2D axes
area = np.pi * (X[:,1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

#3D axes
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
