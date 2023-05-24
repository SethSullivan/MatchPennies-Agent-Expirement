import numpy as np
import analysis_utilities as au
import pingouin as pg
from itertools import combinations
import plot_functions as pf
import data_visualization as dv
import matplotlib.pyplot as plt
wheel  = dv.ColorWheel()
class Statistics():
    def __init__(self,df,experiment,num_subjects,num_blocks,num_trials,trial_block_titles,
                 f1_xticklabels,f2_xticklabels,f1_xlabel,f2_xlabel):
        self.experiment         = experiment
        self.df                 = df
        self.num_subjects       = num_subjects
        self.num_blocks         = num_blocks
        self.num_trials         = num_trials
        self.condition_nums     = np.arange(0,self.num_blocks+1,1).astype(str)
        self.trial_block_titles = trial_block_titles
        self.f1_xticklabels     = f1_xticklabels
        self.f1_xlabel          = f1_xlabel
        self.f2_xlabel          = f2_xlabel
        self.f2_xticklabels     = f2_xticklabels
        
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
        
    def df_to_array(self,df_col):
        ans = np.array(df_col).reshape(self.num_subjects,self.num_blocks)
        return ans
    
    
    def run_statistics(self,dv,alternative='two-sided',M=1e7):
        self.M = M
        
        # Check to make sure that factor 1 is the means for exp1 or the incorrect punishment for exp2
        assert self.df['Factor 1'].str.contains('1000').any() or self.df['Factor 1'].str.contains('-1 Inc').any()
        
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
            f1_collapse_pvals_dict,f1_collapse_cles_dict,\
            f2_collapse_pvals_dict,f2_collapse_cles_dict = self.collapsed_bootstrap(metric,alternative = alternative)
                    
            d = dict(zip(['f1pvals','f1eff','f2pvals','f2eff'],[f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict]))
            return anova,[f1_collapse_pvals_dict,f1_collapse_cles_dict,f2_collapse_pvals_dict,f2_collapse_cles_dict]


    def pairwise_bootstrap(self,data,condition_nums=None,alternative='two-sided'):
        
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
            pvals[c] = au.bootstrap(data[:,i],data[:,j],paired=True,M=self.M,alternative=alternative)
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
    
    def plot_collapse(self,statistics,metric,collapse_factor, title, ylab,title_pad = 10,statline_ypos = None,
                         h=5,num_yticks = None, ylocs=None,lims=True):
        
        if collapse_factor == 'f1':
            condition_nums = self.f1_condition_nums
            xticklabs      = self.f1_xticklabels
            xlab           = self.f1_xlabel
            combos         = self.f1_collapse_combos
        if collapse_factor == 'f2':
            condition_nums = self.f2_condition_nums
            xticklabs      = self.f2_xticklabels
            xlab           = self.f2_xlabel
            combos         = self.f2_collapse_combos

                
        metric = self.collapse_across(metric,combos)
        n = metric.shape[-1]
        w= 9.5
        h = w
        bw = 0.3*w/n # box widths of all boxes combined takes up 30% of the width
        print(bw)
        axmain,ax = pf.make_figure_panel(figsize=(w,h),inset_size=(1.3,0.9,w-1.4,h-1.3))
        axmain.set_aspect(0.6)
        # ax.set_aspect(1.5)
        xlocs = np.linspace(0,w,num=n)
        shift = np.min(metric)*0.12
        if num_yticks is None:
            num_yticks = 8
        if ylocs is None:
            ylocs = np.linspace(np.min(metric),np.max(metric),num_yticks)
        if statline_ypos == None:
            start = np.max(metric) + shift//n
            end = np.max(metric) + shift*0.3
            statline_ypos = np.linspace(start,end,n)
        pf.multi_boxplot(ax,metric,xlocs=xlocs,box_width = bw,colors = wheel.seth_blue)
        dv.jitter_array(ax=ax,x_positions=xlocs,data_list=metric.T, noise_scale=0.15, include_mean = True, circle_size=70)
        condition_locs = self.create_combos(condition_nums)
        if self.experiment == 'Exp1' and collapse_factor=='f1':
                condition_locs[-1],condition_locs[-2] = condition_locs[-2],condition_locs[-1] 
        i=-1
        for c in condition_locs:
            i+=1
            a = int(c[0])
            b = int(c[1])
            dv.stat_annotation(ax,xlocs[a],xlocs[b],statline_ypos[i],p_val=statistics[0][c],cles=statistics[1][c],fontsize=12,h=h)
            statline_ypos += shift

        ax.set_xticks(xlocs),ax.set_yticks(ylocs)
        ax.set_xticklabels(xticklabs)
        if lims:
         
            ax.set_xlim(min(xlocs)-2*bw,max(xlocs)+2*bw)
            ax.set_ylim(min(ylocs),max(ylocs)+2*shift)
            

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=title_pad)
        axmain.set_axis_off()
        plt.show()
                