
# coding: utf-8

# In[10]:

import pymc3 as pm
from sklearn.decomposition import PCA
import pandas as pd
from time import time
from sklearn.cluster import KMeans


# In[2]:

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


# In[209]:

from scGeneFit.functions import *

get_ipython().magic('matplotlib inline')
import numpy as np
np.random.seed(0) 
from sklearn.preprocessing import normalize


# In[4]:

import theano


# ### Load data

# In[204]:

from sklearn.neighbors import NearestCentroid
clf=NearestCentroid()

def performance(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)


# In[220]:

def marker_results(cite_seq,num_mgenes):
    '''
    cite_seq: boolean, are we using citseq data (otherwise us zeisel)
    num_genes: number of marker genes candidates to test
    
    output:
    sample_betas: an array of size # of cell types, of numpy arrays, which contain the betas for that cell type 
                    corresponding to the marker genes identified for that cell type 
    '''
    if cite_seq:
        [data, labels, names]= load_example_data("CITEseq")
        
        #meta clusters
        kmeans = KMeans(n_clusters=8, random_state=42).fit(data)
        flevel = kmeans.labels_
        
        #only take genes with highest variance
        var = normalize(data, norm="l1").var(axis = 0)
        ind = np.argpartition(var, -num_mgenes)[-num_mgenes:]
        inp = normalize(data, norm="l1")[:,ind] * 1000
        
        sample_betas = []
        #regression for each cell type
        for lab in np.unique(labels):
            obs = np.array(labels == lab).astype(int)
            sb = get_bayes_markers(obs,inp,flevel,num_mgenes)
            sample_betas.append(sb)
            print("Got marker genes for cell type {}".format(lab))
            
    else:
        [data, labels, names]= load_example_data("zeisel")
        
        #meta clusters
        flevel = labels[0] - 1
        
        #only take genes with highest variance
        var = normalize(data, norm="l1").var(axis = 0)
        ind = np.argpartition(var, -num_mgenes)[-num_mgenes:]
        inp = normalize(data, norm="l1")[:,ind] * 1000
        
        sample_betas = []
        #regression for each cell type
        for lab in np.unique(labels[1]):
            obs = np.array(labels[1] == lab).astype(int)
            sb = get_bayes_markers(obs,inp,flevel,num_mgenes)
            sample_betas.append(sb)
            print("Got marker genes for cell type {}".format(lab))
    
    return sample_betas
    


# In[221]:

def get_bayes_markers(obs,inp,flevel,mgenes):
    basic_model = pm.Model()
    with basic_model:
        #priors for random intercept
        mu_a = pm.Normal('mu_a', mu=0., sigma=2)
        sigma_a = pm.HalfNormal('sigma_a', 1)
        # Intercept for each cluster, distributed around group mean mu_a
        # Above we just set mu and sd to a fixed value while here we
        # plug in a common group distribution for all a and b (which are
        # vectors of length n_counties).
        num_levels = len(np.unique(flevel))
        alpha = pm.Normal('alpha', mu=mu_a, sigma=sigma_a,shape = num_levels)

        # Priors for unknown model parameters
        #beta = pm.Beta("beta", alpha=1/2, beta = 1/2, shape=20)
        beta = pm.Laplace("beta", mu=0, b = 1, shape=mgenes)
        #beta = pm.Normal("beta", mu=0, sigma = 0.5, shape=30)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Bernoulli("Y_obs", logit_p=alpha[flevel] + inp @ beta, observed=obs)
        
        map_estimate = pm.find_MAP(model=basic_model)
        map_betas = np.where(map_estimate['beta'] > 0)[0]
        #print(map_betas)
    with basic_model:
        # draw 500 posterior samples
        trace = pm.sample(200)
    df = az.summary(trace, round_to=2)
    #print(df)
    betas = df.iloc[9:-1,:]
    #betas with significant values
    sample_betas = betas[betas["hpd_3%"] > 0].index.values
    #return the actual number (0,mgenes-1) of the genes that were significant
    offset = num_levels + 1
    beta_values =  np.where(df.index.isin(sample_betas))[0] - offset
    print(beta_values)
    return beta_values


# In[186]:

# citeseq_betas_20 = marker_results(True,20)
# citeseq_betas_25 = marker_results(True,25)
# citeseq_betas_30 = marker_results(True,30)


# In[ ]:

zeisel_betas_30 = marker_results(False,30)
zeisel_betas_40 = marker_results(False,40)
zeisel_betas_50 = marker_results(False,50)

# In[217]:
print(np.concatenate(zeisel_betas_30))
print(np.concatenate(zeisel_betas_40))
print(np.concatenate(zeisel_betas_50))


def z_accuracy_comp(res,mgenes):
    
    mgenes_list = list(set(np.concatenate(res)))
    [data, labels, names]= load_example_data("zeisel")
    #meta clusters
#     kmeans = KMeans(n_clusters=8, random_state=42).fit(data)
#     flevel = kmeans.labels_

    #only take genes with highest variance
    var = normalize(data, norm="l1").var(axis = 0)
    ind = np.argpartition(var, -mgenes)[-mgenes:]
    inp = normalize(data, norm="l1")[:,ind] * 1000
    inp = inp[:,mgenes_list]
    
    num_markers=mgenes
    method='centers'
    redundancy=0.25

    markers= get_markers(data, labels, num_markers, method=method, redundancy=redundancy)

    accuracy=performance(data[:2100], labels[1][:2100], data[2100:], labels[1][2100:], clf)
    accuracy_markers=performance(data[:2100,markers], labels[1][:2100], data[2100:,markers], labels[1][2100:], clf)
    our_accuracy = performance(inp[:2100],labels[1][:2100],inp[2100:],labels[1][2100:],clf)
    print("Accuracy (whole data,", d, " markers): ", accuracy)
    print("Accuracy (scGenefit) (selected", num_markers, "markers)", accuracy_markers)
    print("Accuracy (MAGE-SELECT) (selected", len(mgenes_list), "markers)", our_accuracy)


# In[216]:

z_accuracy_comp(zeisel_betas_30,30)
z_accuracy_comp(zeisel_betas_40,40)
z_accuracy_comp(zeisel_betas_50,50)


