import numpy as np
import analysis_utilities as au
import pingouin as pg
from itertools import combinations
import plot_functions as pf
import data_visualization as dv
import matplotlib.pyplot as plt
import copy
wheel  = dv.ColorWheel()
class Statistics():
    def __init__(self,df,experiment,num_subjects,num_blocks,num_trials,trial_block_titles,xlab,
                 f1_xticklabels,f2_xticklabels,f1_xlabel,f2_xlabel,M):
        self.experiment         = experiment
        self.df                 = df
        self.num_subjects       = num_subjects
        self.num_blocks         = num_blocks
        self.num_trials         = num_trials
        self.condition_nums     = np.arange(0,self.num_blocks,1).astype(str)
        self.trial_block_titles = trial_block_titles
        self.xlabel             = xlab
        self.f1_xticklabels     = f1_xticklabels
        self.f1_xlabel          = f1_xlabel
        self.f2_xlabel          = f2_xlabel
        self.f2_xticklabels     = f2_xticklabels
        self.M_init                  = M
        
        assert self.experiment == 'Exp1' or self.experiment == 'Exp2'

        if self.experiment == 'Exp1':
            self.f1_collapse_combos = ['01','23','45'] # Combinations to run in pairwise bootstrap for MEAN factor
            self.f2_collapse_combos = ['024','135'] # Combinations to run in pairwise bootstrap for SD factor
            self.f1_condition_nums = ['0','1','2'] # Call the condition numbers 0 1 2 for plotting
            self.f2_condition_nums = ['0','1']

        if self.experiment == 'Exp2':
            self.f1_collapse_combos = ['02','13'] # Combinations to run in pairwise bootstrap for INCORRECT PUNISHMENT factor, collapse across indecision factor
            self.f2_collapse_combos = ['01','23'] # Combinations to run in pairwise bootstrap for INDECISION PUNISHMENT factor, collapse across incorrect factor
            self.f1_condition_nums = ['0','1']
            self.f2_condition_nums = ['0','1']
        # self.run_statistics_all()
        self.anova_dict = {}
        
    def df_to_array(self,df_col):
        ans = np.array(df_col).reshape(self.num_subjects,self.num_blocks)
        return ans
    
    
    def run_statistics(self,dv,change_m = None,alternative='two-sided',test='mean'):
        if change_m is None:
            self.M = copy.deepcopy(self.M_init)
        else:
            self.M = change_m
            
        self.test = test
        # Check to make sure that factor 1 is the means for exp1 or the incorrect punishment for exp2
        assert self.df['Factor 1'].str.contains('1000').any() or self.df['Factor 1'].str.contains('-1 Inc').any() or self.df['Factor 1'].str.contains('Reaction').any()
        
        self.anova = pg.rm_anova(data=self.df, dv=dv, within=['Factor 1','Factor 2'], subject='Subject', detailed=True)
        if dv not in self.anova_dict.keys():
            self.anova_dict.update({dv:self.anova})
        
        assert self.anova['Source'][2] == 'Factor 1 * Factor 2' # Make sure row 2 is the interaction 
        
        metric = self.df_to_array(self.df[dv])
        
        #* Don't collapse 
        if self.anova['p-GG-corr'][2] <0.05:
            print('Significant interaction, doing pairwise bootstraps for each condition...')
            self.collapse_factor = [None]
            pval_dict,cles_dict = self.pairwise_bootstrap(metric,alternative=alternative)
            return self.anova,[pval_dict,cles_dict]

        #* Collapse
        else:
            if self.anova['p-GG-corr'][2] <0.1:
                print('Interaction significance close')
            self.collapse_factor = ['f1','f2']
            print('Non-significant interaction, collapsing across conditions...')
            f1_collapse_pvals_dict,f1_collapse_cles_dict,\
            f2_collapse_pvals_dict,f2_collapse_cles_dict = self.collapsed_bootstrap(metric,alternative = alternative)
                    
            d = dict(zip(['f1pvals','f1eff','f2pvals','f2eff'],[f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict]))
            return self.anova,[f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict]


    def pairwise_bootstrap(self,data,condition_nums=None,alternative='two-sided',
                           **kwargs):
        def _check_parity(combo):
            a = int(combo[0])
            b = int(combo[1])
            if a%2 == b%2:
                return True
            else:
                return False
            
        #* Need these to be able to be generated incase I don't want to run an anova 
        # This is a use case for the mini reaction time experiment
        if not hasattr(self,'M'):
            self.M = kwargs.get('M')
        if not hasattr(self,'test'):
            self.test= kwargs.get('test')    
        
        #* If not collapsing, then go through every combination
        if condition_nums is None: 
            if self.experiment == 'Exp1':
                condition_nums = ['0','1','2','3','4','5']
                # Only take the even conditions together and the odd conditions together
                combos_ = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)] # Creates list of unique combos, order doesn't matter
                combos = [c for c in combos_ if _check_parity(c)]
            elif self.experiment == 'Exp2':
                condition_nums = ['0','1','2','3']
                combos = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)] # Creates list of unique combos, order doesn't matter
        else:
            combos = ["".join(map(str, comb)) for comb in combinations(condition_nums, 2)] # Creates list of unique combos, order doesn't matter

        c=-1
        pvals = np.empty((len(combos)))
        cles1 = np.empty((len(combos)))
        cles2 = np.empty((len(combos)))
        for combo in combos:
            c+=1
            i = int(combo[0])
            j = int(combo[1])
            pvals[c] = au.bootstrap(data[:,i],data[:,j],paired=True,M=self.M,alternative=alternative,test=self.test)
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
        f1_collapse_metric = self.collapse_across(metric,self.f1_collapse_combos) # Collapsing across f2 to get the f1 combined
        # Bootstrap collapsing across f2 (f1_collapse)
        f1_collapse_pvals_dict,f1_collapse_cles_dict = self.pairwise_bootstrap(f1_collapse_metric,alternative=alternative,condition_nums=self.f1_condition_nums)
        
        f2_collapse_metric = self.collapse_across(metric,self.f2_collapse_combos)
        # Bootstrap collapsing across f1 (f2 collapse)
        f2_collapse_pvals_dict,f2_collapse_cles_dict = self.pairwise_bootstrap(f2_collapse_metric,alternative=alternative,condition_nums=self.f2_condition_nums)
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
    
    def create_combos(self,num_list):
        return ["".join(map(str, comb)) for comb in combinations(num_list, 2)] 
    
    def plot_all_conditions(self):
        pass
    
    def plot(self,statistics,metric_name, title, ylab,title_pad = 10,statline_ypos = None,
                         h=5,num_yticks = None, ylocs=None,lims=True,cut_pvals = False,box_colors = wheel.seth_blue,):
        #* Set values for the factor that we're collapsing across
        if self.anova['p-GG-corr'][0] >0.05:
            print('!! Factor 1 is not significant !!')
        if self.anova['p-GG-corr'][1] >0.05:
            print('!! Factor 2 is not significant !!')
            
        for collapse_factor in self.collapse_factor:            
            if collapse_factor != None: # Check to make sure that if we collapsed, we didn't accidentally select specific conditions cuz we want to see all of them
                print('Collapsing, changed select conditions to be All')
                select_conditions = ['All']  
            else:
                if self.experiment == 'Exp1':
                    select_conditions = ['odd','even']
                elif self.experiment == 'Exp2':
                    select_conditions = ['All']
                    
            for conditions in select_conditions:
                metric = self.df_to_array(self.df[metric_name]) # Get metric as an array
                statline_y = statline_ypos 
                
                if True:
                    if collapse_factor == 'f1':
                        condition_nums = self.f1_condition_nums
                        xticklabs      = self.f1_xticklabels
                        xlab           = self.f1_xlabel
                        combos         = self.f1_collapse_combos
                        stat_pval_id   = 0
                        stat_cles_id   = 1
                    elif collapse_factor == 'f2':
                        condition_nums = self.f2_condition_nums
                        xticklabs      = self.f2_xticklabels
                        xlab           = self.f2_xlabel
                        combos         = self.f2_collapse_combos
                        stat_pval_id   = 2
                        stat_cles_id   = 3
                    else:
                        condition_nums = self.condition_nums
                        xticklabs = self.trial_block_titles
                        xlab      = self.xlabel
                        combos    = self.create_combos(condition_nums)
                        stat_pval_id   = 0
                        stat_cles_id   = 1
                    
                if collapse_factor != None: 
                    metric = self.collapse_across(metric,combos)   
                
                #* Get plot constants
                if True:
                    n = metric.shape[-1]
                    width= 8.5 + n/2
                    height = width
                    bw = 0.4*width/n # box widths of all boxes combined takes up 30% of the width
                    axmain,ax = pf.make_figure_panel(figsize=(width,h),inset_size=(1.3,0.9,width-1.4,height-1.3))
                    axmain.set_aspect(0.6)
                    # ax.set_aspect(1.5)
                    xlocs = np.linspace(0,width,num=n)
                    if np.max(metric)<=100:
                        shift = 8
                    elif np.max(metric)<=500:
                        shift=26
                    else:
                        shift=60
                        
                    if num_yticks is None:
                        num_yticks = 8
                        
                    if ylocs is None:
                        ylocs = np.linspace(np.min(metric),np.max(metric),num_yticks)
                #* get xlocs and labels if we only want certain conditions
                if True:
                    xlocs_bp = copy.deepcopy(xlocs)
                    xlocs_sa = copy.deepcopy(xlocs)
                    if conditions == 'odd':
                        metric = metric[:,::2]
                        bw = 0.4*width/metric.shape[-1]
                        condition_nums = condition_nums[::2]
                        xlocs_bp = xlocs_bp[::2] + (width/(2*n)) # CENTER the xlocs
                        xlocs_sa = xlocs_sa + (width/(2*n))           
                        xticklabs= xticklabs[::2]
                    elif conditions == 'even':
                        metric = metric[:,1::2]
                        bw = 0.4*width/metric.shape[-1]
                        condition_nums = condition_nums[1::2]
                        xlocs_bp = xlocs_bp[1::2] - (width/(2*n)) # CENTER the xlocs
                        xlocs_sa = xlocs_sa - (width/(2*n))           
                        xticklabs= xticklabs[1::2]
                    elif conditions == 'All':
                        pass
                    else:
                        raise KeyError('select_conditions must be All, even, or odd')
                
                _,B = pf.multi_boxplot(ax,metric,xlocs=xlocs_bp,box_width = bw,colors = box_colors,)
                dv.jitter_array(ax=ax,x_positions=xlocs_bp,data_list=metric.T, noise_scale=0.15, include_mean = True, circle_size=50)
                
                #* Get condition xlocs and plot stat annotation 
                if True:
                    condition_locs = self.create_combos(condition_nums)
                    # Swap condition locations so the long pval is on top (between 0 and 2)
                    if self.experiment == 'Exp1' and (collapse_factor=='f1' or collapse_factor == None):
                            condition_locs[-1],condition_locs[-2] = condition_locs[-2],condition_locs[-1]         
            
                    top_whisk = np.array([item.get_ydata()[0] for item in B['caps']]) # Get the top whiskers of all the plots
                    if statline_y is None:
                        statline_y = np.max(top_whisk) + shift//3 # put the stat annotation a little above the top whisker
                    
                    #* Plot the stat annotations
                    for c in condition_locs:
                        a = int(c[0])
                        b = int(c[1])
                        # Skip the annotation if the pvalue is above 0.1
                        if collapse_factor is None and statistics[stat_pval_id][c]>0.1 and cut_pvals == True:
                            continue
                        else:
                            dv.stat_annotation(ax,xlocs_sa[a],xlocs_sa[b],statline_y,p_val=statistics[stat_pval_id][c],cles=statistics[stat_cles_id][c],
                                            fontsize=12,h=h)
                            statline_y += shift

                ax.set_xticks(xlocs_bp),ax.set_yticks(ylocs)
                ax.set_xticklabels(xticklabs)
                if lims:
                    ax.set_xlim(min(xlocs)-2*bw,max(xlocs)+2*bw)
                    ax.set_ylim(min(ylocs),max(ylocs))
                
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
                ax.set_title(title, pad=title_pad)
                axmain.set_axis_off()
                plt.show()
                