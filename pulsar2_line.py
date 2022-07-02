#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bilby
import matplotlib.pyplot as plt
import numpy as np


# In[15]:


df=pd.read_csv("C:\\Users\\AYUSH PANDEY\\Desktop\\pulsar2.csv")


# In[26]:


import numpy as np
x=df['time']
y=df['y']
yerr=df['yerr']


# In[17]:


def model_array(x, m, p,c,d):
    model_array=[]
    for i in x:
        if(i<=59232.91566):
            model = m*i + p
            model_array.append(model)
        else:
            model= c*i + d
            model_array.append(model)
    model_array= np.array(model_array) 
    return model_array


# In[18]:


likelihood = bilby.likelihood.GaussianLikelihood(x, y, model_array, yerr)


# In[40]:


priors = dict()
priors["m"] = bilby.core.prior.Uniform(-10**-5, 10**-5, "m")
priors["p"] = bilby.core.prior.Uniform(-10**-1, 10**-1, "p")
priors["c"] = bilby.core.prior.Uniform(-10**-3, -10**-5, "c")
priors["d"] = bilby.core.prior.Uniform(1, 100, "d")


# In[43]:


injection_parameters = dict(m=1.10513821e-06, p=-4.50155384e-02, c=-3.34641245e-04, d=1.98813802e+01)
label = "linear_regression"
outdir= "outdir7"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# In[44]:


result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# Finally plot a corner plot: all outputs are stored in outdir
result.plot_corner()


# In[25]:





# In[ ]:




