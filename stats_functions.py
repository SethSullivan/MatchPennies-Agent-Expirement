import numpy as np
import analysis_utilities as au
import pingouin as pg
from itertools import combinations

# class Statistics():
#     def __init__(self,df,dv,arr):
#         self.experiment = experiment
        
#         self.run_statistics(df,dv,arr)
        
def run_statistics(df,dependent_variable_name,arr,alternative='two-sided'):
    
    # Check to make sure that factor 1 is the means for exp1 or the incorrect punishment for exp2
    assert df['Factor 1'].str.contains('1000').any() or df['Factor 1'].str.contains('-1 Inc').any()
    
    anova = pg.rm_anova(data=df, dv=dependent_variable_name, within=['Factor 1','Factor 2'], subject='Subject', detailed=True)
    
    assert anova['Source'][2] == 'Factor 1 * Factor 2' # Make sure row 2 is the interaction 
    
    # Don't collapse 
    if anova['p-GG-corr'][2] <0.05:
        print('Significant interaction, doing pairwise bootstraps for each condition...')
        pval_dict,cles_dict = pairwise_bootstrap(arr,alternative=alternative)
        return anova,[pval_dict,cles_dict]

    # Collapse
    else:
        print('Non-significant interaction, collapsing across conditions...')
        mean_collapse_pvals_dict,mean_collapse_cles_dict,sd_collapse_pvals_dict,sd_collapse_cles_dict = collapsed_bootstrap(arr)
        return anova,[mean_collapse_pvals_dict,mean_collapse_cles_dict,sd_collapse_pvals_dict,sd_collapse_cles_dict]
    

def pairwise_bootstrap(data,paired = True,alternative='two-sided',condition_nums = None,M=1e7):
    if condition_nums is None:
        condition_nums = ['0','1','2','3','4','5']
        
    combos = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)] # Creates list of unique combos, order doesn't matter
    c=-1
    pvals = np.empty((len(combos)))
    cles1 = np.empty((len(combos)))
    cles2 = np.empty((len(combos)))
    for combo in combos:
        c+=1
        i = int(combo[0])
        j = int(combo[1])
        pvals[c] = au.bootstrap(data[:,i],data[:,j],paired=paired,M=M,alternative='two-sided')
        cles1[c] = au.cles(data[:,i],data[:,j],paired=paired) 
        cles2[c] = au.cles(data[:,j],data[:,i],paired=paired) 
    # Create array and do holm bonferroni
    check,pvals_corrected = pg.multicomp(pvals=pvals,method='holm')
    pvals_corrected = au.holmbonferroni_correction(pvals)
    pval_dict = {}
    pval_dict = dict(zip(combos, pvals_corrected))

    cles_ = np.maximum(cles1,cles2)
    cles_dict = {}
    cles_dict = dict(zip(combos, cles_))
    return pval_dict,cles_dict


def collapsed_bootstrap(metric,paired = True,alternative='two-sided'):
    mean_collapse_combos = ['01','23','45']
    mean_collapse_metric = collapse_across(metric,mean_collapse_combos)
    # Bootstrap collapsing across sds (mean_collapse)
    condition_nums_means = ['0','1','2']
    mean_collapse_pvals_dict,mean_collapse_cles_dict = pairwise_bootstrap(mean_collapse_metric,alternative=alternative,condition_nums=condition_nums_means)
    
    
    sd_collapse_combos = ['024','135']
    sd_collapse_metric = collapse_across(metric,sd_collapse_combos)
    # Bootstrap collapsing across means (sd collapse)
    condition_nums_sds = ['0','1']
    sd_collapse_pvals_dict,sd_collapse_cles_dict = pairwise_bootstrap(sd_collapse_metric,alternative=alternative,condition_nums=condition_nums_sds)
    return mean_collapse_pvals_dict,mean_collapse_cles_dict,sd_collapse_pvals_dict,sd_collapse_cles_dict


def collapse_across(arr,combos):
    ans = []
    if len(combos[0])==2:
        for combo in combos:
            a = int(combo[0])
            b = int(combo[1])
            ans.append(np.concatenate((arr[:,a],arr[:,b])))
    else:
        for combo in combos:
            a = int(combo[0])
            b = int(combo[1])
            c = int(combo[1])
            ans.append(np.concatenate((arr[:,a],arr[:,b],arr[:,c])))
    return_ans = np.array(ans).T
    assert return_ans.shape[0]>15 
    
    return return_ans
            