#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# geopandas
import geopandas as gpd
from shapely.geometry import Point, Polygon
import descartes
import folium
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")


'''Emissions of carbon dioxide are those that come from the burning of fossil fuels.
They consist of gas flaring and carbon dioxide produced during the consumption of solid, liquid, and gas fuels.
Decomposition, oceanic release, and respiration are a few sources of CO2 emissions. 
This investigation focuses on the CO2 emissions that were produced between the years 1980 and 2030 in various nations.
To meet this requirement, two clusters have been formed.
The primary benefit of this activity I is that it allows us to gauge the risk posed by CO2 emissions and take the necessary precautions to keep them under control.
'''

# this function is return the dataframe form world bank
def world_bank(filename):
    
    """it's return the data frame from world bank with filename. """
    
    countries = ["IND","USA","CHN","NGA","NPL","NZL","AUS","AUT","BGD","CAN","AFG",'ALB','DZA','AND','ARG']
    data = wb.data.DataFrame(filename, countries, time = range(2008,2018))
    return data

# print power consumption KWh and population growth%
indicator_code = 'EN.ATM.CO2E.PC'
data = world_bank(indicator_code)

# reset the all the index's
df = data.reset_index()


# # without Normalization

pca = PCA(n_components = 2)
pca.fit(df.iloc[:,1:])



df_pca = pca.transform(df.iloc[:,1:])
df_pca = pd.concat([df.reset_index(drop = True),pd.DataFrame(df_pca)], axis= 1)
df_pca.columns.values[-2: ] = ['component1','component2']


#plot x and y values
x = df_pca['component1']
y = df_pca['component2']

plt.figure()
plt.plot(x, y, "rs", markersize=5)
plt.xlabel("PCA_1")
plt.ylabel("PCA_2")
plt.show()

#ElBOW method use to find clusters
wcss=[] # Within-Cluster-Sum-of-Squares
# Code to get max no of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(df_pca[['component1',"component2"]])
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

xy = df_pca[['component1','component2']]

kmeans = cluster.KMeans(n_clusters=2)
# Fit the data, results are stored in the kmeans object
kmeans.fit(xy) # fit done on x,y pairs
labels = kmeans.labels_
print(labels) # labels is the number of the associated clusters of (x,y)␣

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)
# plot using the labels to select colour
plt.figure(figsize=(5.0,5.0))
col = ["blue", "red","green"]
for l in range(0,2): # loop over the different labels
    plt.plot(x[labels==l], y[labels==l], "go-", markersize=4, color=col[l])
# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
#plt.xlabel("x")
#plt.ylabel("y")
plt.show()

from sklearn.metrics import silhouette_score
score = silhouette_score(xy, kmeans.labels_, metric='euclidean')

# Print the score
print('Silhouetter Score: %.3f' % score)


''' with Normalisation'''
def normalization(data):
    
    """return the normalized data using minmax scaler"""
    
    scaler = MinMaxScaler()
    scaler.fit(data)
    df_norm = scaler.transform(data)

    return df_norm
df_nr = normalization(df.iloc[:,1:])


#pd.DataFrame(df_nr)



# PCA Normalize data
df_nr_pca = PCA(n_components = 2)
df_nr_pca.fit(df_nr)

df_nr_pca = df_nr_pca.transform(df_nr)
df_nr_pca = pd.concat([df.reset_index(drop = True),pd.DataFrame(df_nr_pca)], axis= 1)
df_nr_pca.columns.values[-2: ] = ['component1','component2']


# plot x y values
x = df_nr_pca['component1']
y = df_nr_pca['component2']

# What did we get?
plt.figure()
plt.plot(x, y, "o", markersize=5)
plt.xlabel("pca_1")
plt.ylabel("pca_2")
plt.show()


# ElBOW method use to find clusters
wcss=[] # Within-Cluster-Sum-of-Squares
# Code to get max no of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(df_nr_pca.iloc[:,2:])
    wcss.append(kmeans.inertia_)


plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

# kmeans clustring method
xy = df_nr_pca[['component1','component2']]

kmeans = cluster.KMeans(n_clusters=2)
# Fit the data, results are stored in the kmeans object
kmeans.fit(xy) # fit done on x,y pairs
labels = kmeans.labels_
print(labels) # labels is the number of the associated clusters of (x,y)␣

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)
# plot using the labels to select colour
plt.figure(figsize=(5.0,5.0))
col = ["blue", "red","green"]
for l in range(0,2): # loop over the different labels
    plt.plot(x[labels==l], y[labels==l], ".", markersize=4, color=col[l])
# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


from sklearn.metrics import silhouette_score
score = silhouette_score(xy, kmeans.labels_, metric='euclidean')

'''Print the score'''
print('Silhouetter Score: %.3f' % score)

'''AgglomerativeClustering method'''
ap = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
ap.fit_predict(xy)

ap_labels = ap.labels_
print(ap_labels) # labels is the number of the associated clusters of (x,y)␣

# extract the estimated cluster centres
#cen = ap.cluster_centers_
print(cen)
'''plot using the labels to select colour'''
plt.figure(figsize=(5.0,5.0))
col = ["blue", "red","green"]
for l in range(0,3): # loop over the different labels
    plt.plot(x[labels==l], y[labels==l], "go-", markersize=4, color=col[l])
'''show cluster centres'''
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
'''plt.xlabel("x")'''
'''plt.ylabel("y")'''
plt.show()

from sklearn.metrics import silhouette_score
score = silhouette_score(xy,ap.labels_, metric='euclidean')

''' Print the score'''
print('Silhouetter Score: %.3f' % score)

df_nr_pca['clusters'] = ap.labels_

df_nr_pca = df_nr_pca.rename(columns={'economy':'Countries'})
df_nr_pca

'''create world map from clustring data'''
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
world = world.drop(['pop_est','gdp_md_est'],axis=1)

table = world.merge(df_nr_pca[['Countries','clusters','YR2017']],how='left',left_on=['iso_a3'],right_on=['Countries'])
table = table.dropna()
table.explore(column='Countries', name='clusters')

'''
From the geographical graph, some countries like newzealand,china,india,Afghanistan,Albania, austria, Algeria are segregated into cluster 1 and countries like USA, canada, australia are segregated into 2nd cluster.
'''

''' Fitting'''

import scipy.optimize as opt

'''this function is read dataframe'''
def read_data(name):
    """ return the dataframe,name is a file name."""
    data = pd.read_csv(name)
    return data

data = read_data('emission_fitting.csv') 


'''change column names.'''
data.columns= ['Name','Series Code','Country Name','code','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

'''create new dataframe'''
x = data.columns[4:].T.tolist()
y = data.iloc[0:1,4:].T
y.reset_index(drop=True, inplace=True)
y = y[0].tolist()
dic = {'year':x,"CO2 emission":y}
df = pd.DataFrame(dic)
'''all string values in to flot data type'''
df = df.apply (pd.to_numeric,errors='coerce')
df.dropna()
df.reset_index(drop=True).head()



'''year wise plot CO2 emissions'''
df.plot('year','CO2 emission')
plt.ylabel("CO2 emission")
plt.title('CO2 emission')

plt.show()


# By using the above plot, we are checking the trend of range of co2 emissions from 2004-2018 and we can see the emissions are increasing every year.

'''Logistic function'''



def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f


'''Curve_fit'''

param, covar = opt.curve_fit(logistic, df["year"], df["CO2 emission"], 
                             p0=(3e12, 0.03, 2000.0))

df = df.apply (pd.to_numeric,errors='coerce')
df = df.reset_index(drop=True)

sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)

df["fit"] = logistic(df["year"], *param)

df.plot("year", ["CO2 emission", "fit"])
plt.ylabel("CO2 emission")
plt.title('logistic Curve fit')

plt.savefig('Curve fit.jpg')

'''# From the figure by applying curve fit, we are trying to identify large interval of data for which the 
# proportional growth rate is linear. We can exactly 
# see this phenomenon from the graph for features
# co2 emissions and year for the country India.'''


def err_ranges(x, func, param, sigma):
    """ calculate the error ranges"""
   

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 

year = np.arange(1980, 2031)
print(year)
forecast = logistic(year, *param)


low, up = err_ranges (year, logistic, param, sigma)

plt.figure()
plt.plot(df["year"], df["CO2 emission"], label="CO2 emission")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.ylabel("CO2 emission")
plt.xlabel("Year")
plt.title('CO2 err_range forecast')
plt.legend()

plt.savefig('CO2 forecast.jpg')

print(err_ranges(2030, logistic, param, sigma))

'''We are attempting to identify large intervals of data for which the proportional growth rate is linear by applying curve fit to the data in figure.3 The graph for features of CO2 emissions and year for India allows us to clearly see this phenomenon.
After using K means clustering to divide the world bank data into two clusters, analyzing the clusters, and looking at various correlations between the features, we can determine the forecast of CO2 emissions for the upcoming years based on historical data evidence. This serves as a signal and aids in projecting future country-level increases in CO2 emissions. Results are identical whether normalization is used or not. After applying the curve fit, we calculated the err_ranges, and the results were 1.88919 -2.19177. 
From 1980 to 2030, predictions of CO2 emissions were made.'''




