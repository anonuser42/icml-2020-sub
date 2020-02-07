#!/usr/bin/env python
# coding: utf-8

# In[247]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[248]:


result = pd.read_csv('avg_sat_results.csv', sep=';')
#result = result.iloc[:60]
result['sat_avg'] = result['sat_avg'] / 100
result.head(10)


# In[249]:


def map_filter_sizes(name):
    if '_' not in name:
        x = 'full Filtersize'
    elif '_S' in name:
        x = 'halfed Filtersize'
    elif '_XS' in name:
        x = 'quarter Filtersize'
    elif '_XXS' in name:
        x = 'eigths Filtersize'
    elif '_XXXS' in name:
        x = 'sixteenth Filtersize'
    else:
        x = 'Unknown'
    return x


# In[250]:


result['filter_size'] = result['model'].apply(map_filter_sizes)
result.head(25)


# In[251]:


r1  = result[result['thresh'] == 100]
x = np.unique(result['thresh'].values)
t2 = x[0]
print(t2)
r2 = result[result['thresh'] == t2]
r2.head()


# In[252]:


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import scipy
import matplotlib
matplotlib.rc('text', usetex = True)
params = {'backend': 'ps',
#           'text.latex.preamble': [r'\usepackage{gensymb}'],
          'axes.labelsize': 14, # fontsize for x and y labels (was 10)
          'axes.titlesize': 14,
          'font.size': 10, # was 10
          'legend.fontsize': 10, # was 10
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#plt.rcParams["font.family"] = "serif"
#figsize=(4, 2.5)


# In[253]:


def log(t,a,b,c):
    return a*np.log(t+c)+b#a*(t)+b+c*(t**2)


def plot(sat, acc, names, filter_sizes, title):
    rsat = 1 - sat
    #b, m, n = polyfit(all_average_sat[all_problem == 'CIFAR100'], all_test_acc[all_problem == 'CIFAR100'], 2)
    b10, m10, n10 = polyfit(sat, acc, 2)
    params, cov = curve_fit(log, rsat, acc)
    #b2, m2, n2 = polyfit(sat, acc, 2)
    r2 = r2_score(acc, log(rsat, *params))
    print('r2:\t', r2)
    print('chi2:\t', scipy.stats.chisquare(acc, f_exp=log(rsat, *params)))
    
    fig, ax = plt.subplots()#(10,5))

    cmap = plt.get_cmap('viridis')

    network_legend = [Line2D([0], [0], marker='o', label='VGG11',
                             markerfacecolor='black', 
                             markersize=10),
                     Line2D([0], [0], marker='^',  label='VGG13',
                             markerfacecolor='black', 
                             markersize=10),
                     Line2D([0], [0], marker='s', label='VGG16',
                             markerfacecolor='black', 
                             markersize=10),
                     Line2D([0], [0], marker='p', label='VGG19',
                             markerfacecolor='black', 
                             markersize=10),]

    depth_legend  = [Line2D([0], [0], color=cmap(0.0), lw=4),
                    Line2D([0], [0], color=cmap(0.25), lw=4),
                    Line2D([0], [0], color=cmap(0.5), lw=4),
                    Line2D([0], [0], color=cmap(0.75), lw=4),
                    Line2D([0], [0], color=cmap(1.0), lw=4)]

    x = np.linspace(0, 1, 100)
    #print(x)
    
    
    plt.text(0.6, 0.1,f'$R^2: {round(r2, 3)}$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes,
            fontsize=16)

    colors = {
        'full Filtersize': cmap(0.0),
        'halfed Filtersize' : cmap(0.25),
        'quarter Filtersize' : cmap(0.5),
        'eigths Filtersize': cmap(0.75),
        'sixteenth Filtersize': cmap(1.0)
    }
    
    def network_depth(model):
        mapper = {
        'VGG11': 'o',
        'VGG13': '^',
        'VGG16': 's',
        'VGG19': 'p'
        }
        for key, val in mapper.items():
            if key in model:
                return val
        return None

    filter_legend = []



    for i in range(len(sat)):
        ax.scatter(x=sat[i], y=acc[i], color=colors[filter_sizes[i]], marker=network_depth(names[i]))
    #plt.plot(x, n*(x**2)+m*x+b, c='orange', label='CIFAR100 regression parabola')
    #ax.plot(x, n2*(x**2)+m2*x+b2, c='green')
    ax.plot(x, log(1.0-x, *params), c='grey')

    l1 = plt.legend(network_legend, ['VGG11', 'VGG13', 'VGG16', 'VGG19'], loc=1, prop={'size': 11})
    ax = plt.gca().add_artist(l1)
    plt.legend(depth_legend, ['full Filtersize', 
                               'halfed Filtersize', 
                               'quarter Filtersize', 
                              'eigths Filtersize',
                              'sixteenth Filtersize'], loc=3, prop={'size': 11})
    plt.xlim((0,1))
   # plt.ylim((0, 1))

    #for i, xy in enumerate(zip(all_average_sat, all_test_acc)):
    #    plt.annotate(all_names[i], xy)

    #plt.legend(prop={'size': 13})
    plt.grid()




    #plt.title('Average Layer Saturation at Training Time versus Test Accuracy')
    plt.xlabel('Saturation', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=16)

    plt.savefig(
        f"{title}.eps",
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )
    return x


# ## CIFAR10, 100 and CatVsDog with all data

# In[254]:


ds = 'Cifar10'
x = plot(r2[r2['dataset'] == ds]['sat_avg'].values, r1[r1['dataset'] == ds]['accs'].values, r2[r2['dataset'] == ds]['model'].values, r2[r2['dataset'] == ds]['filter_size'].values, 'avgsat_cifar10_raw')
print(x)


# In[255]:


ds = 'Cifar100'
plot(r2[r2['dataset'] == ds]['sat_avg'].values, r1[r1['dataset'] == ds]['accs'].values, r2[r2['dataset'] == ds]['model'].values, r2[r2['dataset'] == ds]['filter_size'].values, 'avgsat_cifar100_raw')


# In[256]:


ds = 'CatVsDog'
plot(r2[r2['dataset'] == ds]['sat_avg'].values, r1[r1['dataset'] == ds]['accs'].values, r2[r2['dataset'] == ds]['model'].values, r2[r2['dataset'] == ds]['filter_size'].values, 'avgsat_cvd_raw')


# ## CIFAR10, 100 and CatVsDog with averaged duplicate runs

# In[257]:


r1_g = r1.groupby(['thresh', 'model', 'dataset', 'filter_size']).mean().reset_index()
r2_g = r2.groupby(['thresh', 'model', 'dataset', 'filter_size']).mean().reset_index()

r1_g.head(10)


# In[258]:


r1 = r1_g
r2 = r2_g


# In[259]:


ds = 'Cifar10'
plot(r2[r2['dataset'] == ds]['sat_avg'].values, r1[r1['dataset'] == ds]['accs'].values, r2[r2['dataset'] == ds]['model'].values, r2[r2['dataset'] == ds]['filter_size'].values, 'avgsat_cifar10_avg')


# In[260]:


ds = 'Cifar100'
plot(r2[r2['dataset'] == ds]['sat_avg'].values, r1[r1['dataset'] == ds]['accs'].values, r2[r2['dataset'] == ds]['model'].values, r2[r2['dataset'] == ds]['filter_size'].values, 'avgsat_cifar100_avg')


# In[261]:


ds = 'CatVsDog'
plot(r2[r2['dataset'] == ds]['sat_avg'].values, r1[r1['dataset'] == ds]['accs'].values, r2[r2['dataset'] == ds]['model'].values, r2[r2['dataset'] == ds]['filter_size'].values, 'avgsat_cvd_avg')


# In[ ]:





# In[ ]:





# In[ ]:




