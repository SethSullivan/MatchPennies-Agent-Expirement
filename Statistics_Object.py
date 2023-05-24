import numpy as np
import analysis_utilities as au
import pingouin as pg
from itertools import combinations

class Statistics():
    def __init__(self,df,experiment,num_subjects,num_blocks,num_trials):
        self.experiment = experiment
        self.df = df
        self.num_subjects = num_subjects
        self.num_blocks = num_blocks
        self.num_trials = num_trials
        self.condition_nums = np.arange(0,self.num_blocks+1,1).astype(str)
        # self.run_statistics_all()
        
    def df_to_array(self,df_col):
        ans = np.array(df_col).reshape(self.num_subjects,self.num_blocks)
        return ans
    
    
    def run_statistics(self,dv,alternative='two-sided'):
        
        # Check to make sure that factor 1 is the means for exp1 or the incorrect punishment for exp2
        assert self.df['Factor 1'].str.contains('1000').any() or self.df['Factor 1'].str.contains('-1 Inc').any()
        assert self.experiment == 'Exp1' or self.experiment == 'Exp2'
        
        anova = pg.rm_anova(data=self.df, dv=dv, within=['Factor 1','Factor 2'], subject='Subject', detailed=True)
        
        assert anova['Source'][2] == 'Factor 1 * Factor 2' # Make sure row 2 is the interaction 
        
        metric = self.df_to_array(self.df[dv])
        
        #* Don't collapse 
        if anova['p-GG-corr'][2] <0.05:
            if anova['p-GG-corr'][2] <0.1:
                print('Interaction significance close')
            print('Significant interaction, doing pairwise bootstraps for each condition...')
            pval_dict,cles_dict = self.pairwise_bootstrap(metric,alternative=alternative)
            return anova,[pval_dict,cles_dict]

        #* Collapse
        else:
            print('Non-significant interaction, collapsing across conditions...')
            f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict = self.collapsed_bootstrap(metric,alternative = alternative)
            return anova,[f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict]
        

    def pairwise_bootstrap(self,data,condition_nums=None,alternative='two-sided',M=1e7):
        
        #* If not collapsing, then go through every combination
        if condition_nums is None: 
            if self.experiment == 'Exp1':
                condition_nums = ['0','1','2','3','4','5']
            elif self.experiment == 'Exp2':
                condition_nums = ['0','1','2','3']

        combos = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)] # Creates list of unique combos, order doesn't matter
        c=-1
        pvals = np.empty((len(combos)))
        cles1 = np.empty((len(combos)))
        cles2 = np.empty((len(combos)))
        for combo in combos:
            c+=1
            i = int(combo[0])
            j = int(combo[1])
            pvals[c] = au.bootstrap(data[:,i],data[:,j],paired=True,M=M,alternative=alternative)
            cles1[c] = au.cles(data[:,i],data[:,j],paired=True) 
            cles2[c] = au.cles(data[:,j],data[:,i],paired=True) 
            
        # Create array and do holm bonferroni
        check,pvals_corrected = pg.multicomp(pvals=pvals,method='holm')
        pvals_corrected = au.holmbonferroni_correction(pvals)
        pval_dict = {}
        pval_dict = dict(zip(combos, pvals_corrected))

        cles_ = np.maximum(cles1,cles2)
        cles_dict = {}
        cles_dict = dict(zip(combos, cles_))
        return pval_dict,cles_dict

    def collapsed_bootstrap(self,metric,alternative='two-sided'):
        if self.experiment == 'Exp1':
            f1_collapse_combos = ['01','23','45'] # Combinations to run in pairwise bootstrap for MEAN factor
            f2_collapse_combos = ['024','135'] # Combinations to run in pairwise bootstrap for SD factor
            f1_condition_nums = ['0','1','2'] # Call the condition numbers 0 1 2 for plotting
            f2_condition_nums = ['0','1']
        if self.experiment == 'Exp2':
            f1_collapse_combos = ['02','13'] # Combinations to run in pairwise bootstrap for INCORRECT PUNISHMENT factor, collapse across indecision factor
            f2_collapse_combos = ['01','23'] # Combinations to run in pairwise bootstrap for INDECISION PUNISHMENT factor, collapse across incorrect factor
            f1_condition_nums = ['0','1']
            f2_condition_nums = ['0','1']
            
        f1_collapse_metric = self.collapse_across(metric,f1_collapse_combos) # Collapsing across f2 to get the f1 combined
        # Bootstrap collapsing across f2 (f1_collapse)
        f1_collapse_pvals_dict,f1_collapse_cles_dict = self.pairwise_bootstrap(f1_collapse_metric,alternative=alternative,condition_nums=f1_condition_nums)
        
        f2_collapse_metric = self.collapse_across(metric,f2_collapse_combos)
        # Bootstrap collapsing across f1 (f2 collapse)
        f2_collapse_pvals_dict,f2_collapse_cles_dict = self.pairwise_bootstrap(f2_collapse_metric,alternative=alternative,condition_nums=f2_condition_nums)
        return f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict


    def collapse_across(self,arr,combos):
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
                