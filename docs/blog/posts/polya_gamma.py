
---
title: 'polya-gamma'
author: 'jw'
date: 2024-05-23
---



#!/usr/bin/env python
# coding: utf-8

# 

# 
# ## Problem.

# ## Solution.

# In[ ]:


pip install nutpie


# In[ ]:


import numpy as np
import pymc as pm
import arviz as az
import nutpie
import nutpie.compile_pymc


# In[ ]:


def logistic(t):
    return 1/(1+np.exp(-t))

def generate_data(n_subj = 1000, n_item = 15):

    true_θ = np.random.normal(0,1, size=n_subj)
    true_δ = np.random.normal(0,5, size=n_item)

    true_η = np.empty((n_subj, n_item))
    data_y = np.empty((n_subj, n_item))
    for j in range(n_subj):
        for k in range(n_item):
            true_η[j,k] = true_θ[j] - true_δ[k]
            data_y[j,k] = np.random.binomial(1, p=logistic(true_η[j,k]))

    return data_y, true_δ


# In[ ]:


data_y,true_δ = generate_data(n_subj = 1000, n_item = 15)


# In[ ]:


true_δ


# In[ ]:


with pm.Model() as rasch:
    j,k = data_y.shape

    theta = pm.Normal('θ',0,1,shape=(j,1))
    delta_e = pm.Normal('delta_e', 0,5)
    delta_v = pm.HalfCauchy('delta_v',2)
    delta = pm.Normal('δ', delta_e, delta_v, shape = (1,k))

    prob = theta - delta
    y = pm.Bernoulli('y',logit_p = prob, observed=data_y)

    #post_rasch = pm.sample(nuts_sampler='nutpie')


# In[ ]:


az.summary(post_rasch, var_names='δ')


# In[ ]:


with pm.Model() as rasch_polya:
    j,k = data_y.shape

    theta = pm.Normal('θ',0,1,shape=(j,1))
    delta_e = pm.Normal('delta_e', 0,5)
    delta_v = pm.HalfCauchy('delta_v',2)
    delta = pm.Normal('δ', delta_e, delta_v, shape = (1,k))

    prob = theta - delta
    y = pm.PolyaGamma('y',h=1,z=prob, observed=data_y)

    post_rasch_polya = pm.sample()


# In[ ]:





# ## Discussion.

# In[ ]:




