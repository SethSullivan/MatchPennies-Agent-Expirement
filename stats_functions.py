import numpy as np
import analysis_utilities as au
import pingouin as pg

# class Statistics():
#     def __init__(self,df):
#         self.anova_df = self.run_anova()
#     def run_anova(self):
#         pg.rm_anova(data=df, dv='Indecisions', within=['Condition Mean','Condition SD'], subject='Subject', detailed=True)

def pairwise_bootstrap(metric,paired = True):
    pvals = np.empty((15))
    cles1 = np.empty((15))
    cles2 = np.empty((15))
    store_comparisons = []
    d = {}
    c=-1
    for i in range(metric.shape[-1]):
        for j in range(metric.shape[-1]):
            combo1 = str(i)+str(j)
            combo2 = str(j)+str(i)
            if combo1 not in store_comparisons and combo2 not in store_comparisons and i!=j:
                c+=1
                store_comparisons.append(combo1)
                pvals[c] = au.Bootstrap(metric[:,i],metric[:,j],paired=paired,M=1e6) # 1000 (50) to 1000 (150)
                cles1[c] = au.cles(metric[:,i],metric[:,j],paired=paired) # 1000 (50) to 1000 (150)
                cles2[c] = au.cles(metric[:,j],metric[:,i],paired=paired) # 1000 (50) to 1000 (150)
    # Create array and do holm bonferroni
    check,pvals_corrected = pg.multicomp(pvals=pvals,method='holm')
    pval_dict = {}
    keys = store_comparisons
    pval_dict = dict(zip(keys, pvals_corrected))
    
    cles = np.maximum(cles1,cles2)
    cles_dict = {}
    keys = store_comparisons
    cles_dict = dict(zip(keys, cles))
    return pval_dict,cles_dict

def pairwise_collapsed_bootstrap(data1,data2,paired = True):
    pvals = np.empty((15))
    cles1 = np.empty((15))
    cles2 = np.empty((15))
    store_comparisons = []
    d = {}
    c=-1
    
    pvals[c] = au.Bootstrap(metric[:,i],metric[:,j],paired=paired,M=1e6) # 1000 (50) to 1000 (150)
    cles1[c] = au.cles(metric[:,i],metric[:,j],paired=paired) # 1000 (50) to 1000 (150)
    cles2[c] = au.cles(metric[:,j],metric[:,i],paired=paired) # 1000 (50) to 1000 (150)
    # Create array and do holm bonferroni
    check,pvals_corrected = pg.multicomp(pvals=pvals,method='holm')
    pval_dict = {}
    keys = store_comparisons
    pval_dict = dict(zip(keys, pvals_corrected))
    
    cles = np.maximum(cles1,cles2)
    cles_dict = {}
    keys = store_comparisons
    cles_dict = dict(zip(keys, cles))
    return pval_dict,cles_dict