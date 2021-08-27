# -*- coding: utf-8 -*-

# import required libraries for dataframe and visualization
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
# =============================================================================


# import required libraries for clustering
# =============================================================================
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
# =============================================================================

"""
# Pengenalan data
# =============================================================================
retail = pd.read_excel('Online Retail.xlsx')

pd.set_option('display.max_columns', None)
retail.head()
retail.shape
retail.info()
retail.describe() 
# =============================================================================


# Data cleansing
# =============================================================================
df_null = round(100*(retail.isnull().sum())/len(retail), 2)
df_null
retail = retail.dropna()
retail.shape
retail['CustomerID'] = retail['CustomerID'].astype(str)
# =============================================================================


# Data preparation
# =============================================================================
# Monetary
retail['Amount'] = retail['Quantity']*retail['UnitPrice']
rfm_m = retail.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()

# Frequency
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
rfm_f.head()

# Merging the two dataframe (Monetary and Frequency)
rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm.head()

# Recency, converting to datetime 
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format = '%d/%m/%Y %H:%M')

# Finding last transaction date
max_date = max(retail['InvoiceDate'])
max_date

# Compute difference between max date and transaction date
retail['Diff'] = max_date - retail['InvoiceDate']
retail.head()

# Compute last transaction date to get the recency of customers
rfm_p = retail.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()

# Extract number of days only
rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()

# Merge the dataframes to get the final RFM dataframe
rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
rfm.head()

# Outlier analysis of amount frequency and recency
attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')

# Removing (statistical) outliers for Amount
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

# Rescaling the attributes
rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()
# =============================================================================


# Building the model
# =============================================================================
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)

kmeans.labels_

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(range_n_clusters,ssd)
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg.append(silhouette_score(rfm_df_scaled, cluster_labels))
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

sil = [0.5415858652525395,0.5084896296141937,0.4816217519322445,0.46460502083115957,0.4170584389874765,0.41763065866927357,0.40778577162210244]
plt.plot(range_n_clusters, sil, marker= "o")
plt.xlabel('k')
plt.ylabel("Silhouette Coefficient")
plt.title("AVG Silhouette Coefficient")
plt.show()

# Final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)
clusterlabelfinal = kmeans.labels_

rfm['Cluster_Id'] = kmeans.labels_
rfm.head()

#======================Visualisasi hasil cluster=============================
plt.figure(figsize = (8,4))
sns.scatterplot(rfm['Amount'], rfm['Frequency'], hue=clusterlabelfinal, palette=sns.color_palette('husl',3))
plt.title('KMeans with 3 clusters')
plt.show()


sns.boxplot(x='Cluster_Id', y='Amount', data=rfm).set(title='Box plot cluster Monetary')
sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm).set(title='Box plot cluster Frequency')
sns.boxplot(x='Cluster_Id', y='Recency', data=rfm).set(title='Box plot cluster Recency')

mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()

mergings = linkage(rfm_df_scaled, method="average", metric='euclidean')
dendrogram(mergings)
plt.show()

cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels

rfm['Cluster_Labels'] = cluster_labels
rfm.head()

sns.boxplot(x='Cluster_Labels', y='Amount', data=rfm)

sns.boxplot(x='Cluster_Labels', y='Frequency', data=rfm)

sns.boxplot(x='Cluster_Labels', y='Recency', data=rfm)
# =============================================================================
"""

#=============================ATURAN ASOSIASI===============================
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('Online Retail.xlsx')
df.head()

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

basket = (df[df['Country'] =="Singapore"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
























