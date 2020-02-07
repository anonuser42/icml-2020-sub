#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd


# In[51]:


df = pd.read_csv('results_final_random.csv', sep=';')


# In[52]:


df.head(100)


# In[53]:


from matplotlib import pyplot as plt
import numpy as np


# In[54]:


unprojected = df[df['thresh'] == 3.0]
unprojected.head(20)


# In[55]:


all_models = unprojected['model'].values
print(all_models)
print(unprojected[unprojected['model'] == all_models[0]])
acc_dict = {
    model: unprojected[unprojected['model'] == model]['accs'].values[0] for model in all_models
}
loss_dict = {
    model: unprojected[unprojected['model'] == model]['loss'].values[0] for model in all_models
}


# In[56]:


df['diff_acc'] = np.zeros(len(df))
df['diff_loss'] = np.zeros(len(df))


# In[57]:


for model in all_models:
    df.loc[df['model'] == model,'diff_acc'] = df[df['model'] == model]['accs'] / acc_dict[model]
    df.loc[df['model'] == model, 'diff_loss'] = df[df['model'] == model]['loss'] / loss_dict[model] 


# In[58]:


df.head(10)


# In[59]:


buckets = []
means = []
for thresh in np.unique(df['thresh'].values):
    buckets.append(df.loc[df['thresh'] == thresh]['diff_acc'].values)
    means.append(np.mean(buckets[-1]))
print(np.asarray(buckets).shape)
print(buckets[0])
print(means)


# In[63]:


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
import matplotlib
matplotlib.rcParams.update(params)

plt.boxplot(buckets)
plt.plot(list(range(1, len(means)+1)), means, label='mean')
ticks =  np.append(np.around(np.unique(df['thresh'].values*100),1)[:-1], [100.0])
ticks = [tick+'%' for tick in ticks.astype(str)]
print(ticks)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], ticks, rotation=90)
plt.legend()
plt.ylabel('Relative Performance (Accuracy)')
plt.xlabel("\u03B4")
plt.grid()
plt.title('Relative Performance with Random Projections')
plt.savefig(
        f"relative_performance_conv_projection_random.eps",
        format="eps",
        dpi=1000,
        bbox_inches="tight",
    )


# In[ ]:




