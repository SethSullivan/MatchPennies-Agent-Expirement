import matplotlib.pyplot as plt
import numpy as np
import data_visualization as dv
from scipy import stats
wheel = dv.ColorWheel()

class PlottingKwargs:
    def __init__(self,**kwargs):
        self.bw            = kwargs.get('box_width',0.75)
        self.box_color     = kwargs.get('colors',wheel.seth_blue)
        self.xlocs         = kwargs.get('xlocs')
        self.xtick_locs    = kwargs.get('xtick_locs', self.xlocs)
        self.ylocs         = kwargs.get('ylocs')
        self.legend_loc    = kwargs.get('legend_loc','best')
        self.include_means = kwargs.get('include_means')
        self.jitter        = kwargs.get('jitter',True)
        self.line_colors   = kwargs.get('line_colors')
        self.linestyles    = kwargs.get('linestyles')
        self.xticklabels   = kwargs.get('xticklabels')
        self.xlabel        = kwargs.get('xlabel')
        self.ylabel        = kwargs.get('ylabel')
        self.title         = kwargs.get('title')
        self.legend_fontsize = kwargs.get('legend_fontsize',6)
        self.title_fontsize = kwargs.get('title_fontsize',20)
        self.box_lw       = kwargs.get("box_lw", 1.2)
        self.box_width    = kwargs.get("box_width", .75)
        self.whisker_lw   = kwargs.get("whisker_lw", 2.0)

def make_figure_panel(figsize, inset_size, dpi = 100):
    fig,axmain = plt.subplots(dpi = dpi, figsize = figsize)
    
    axmain.set_xlim(0,figsize[0])
    axmain.set_ylim(0,figsize[1])
    axmain.spines.right.set_visible(True)
    axmain.spines.top.set_visible(True)
    ax = axmain.inset_axes(inset_size,transform=axmain.transData)
    
    return axmain,ax

def multi_boxplot(ax, data, xlocs, **kwargs):
    """
    box_lw       = kwargs.get("box_lw", 1.2)
    box_width    = kwargs.get("box_width", .5)
    whisker_lw   = kwargs.get("whisker_lw", 1.2)
    colors       = kwargs.get("colors",   "#0BB8FD")
    x_pos        = kwargs.get("x_pos", 0.25)
    ax           = kwargs.get("ax", None)
    """
    box_lw       = kwargs.get("box_lw", 1.2)
    box_width    = kwargs.get("box_width", .75)
    whisker_lw   = kwargs.get("whisker_lw", 2.0)

    color = kwargs.get("colors",  wheel.seth_blue)
    #box properties
    box_props,whisker_props,cap_props,median_props = {},{},{},{}
    box_props = {"facecolor": "none", "edgecolor" : color, "linewidth": box_lw, "alpha": 1}

    #whisker properties
    whisker_props = {"linewidth" : whisker_lw, "color": color}

    #cap properties
    cap_props = {"linewidth" : whisker_lw, "color": color}

    #median properties
    median_props = {"linewidth" : whisker_lw, "color": color}

    include_means = kwargs.get('include_means')
    '''Make Box Plots'''
    ax.patch.set_alpha(0)
    
    if np.isnan(data).any():
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d,m in zip(data.T, mask.T)]
    else:
        filtered_data = data
    
    bp = ax.boxplot(filtered_data,   positions = xlocs, patch_artist = True,  showfliers = False, 
                boxprops = box_props  , whiskerprops = whisker_props,
                capprops = cap_props, medianprops = median_props, widths = box_width,
                )
    
    # for element in bp:
    #     for patch, color in zip(bp[element],colors):
    #         print(patch)
    #         patch.set_color(color)
    
    return ax,bp

def scatter_with_correlation(ax,xdata,ydata,**kwargs):
    facecolor = kwargs.get('facecolor','none')
    edgecolor = kwargs.get('edgecolor',wheel.seth_blue)
    markersize = kwargs.get('markersize',30)
    alpha = kwargs.get('alpha',1.0)
    lw = kwargs.get('linewidths',1.0)
    correlation = kwargs.get('correlation',True)
    show_spear = kwargs.get('show_spear',True)
    
    xdata = xdata.flatten()
    ydata= ydata.flatten()
    lm = stats.linregress(xdata,ydata)
    x = np.arange(np.nanmin(xdata)-10,np.nanmax(xdata)+10,1)
    y = lm.slope*x + lm.intercept
    spear_r = stats.spearmanr(xdata, ydata)
    ax.scatter(xdata,ydata,c=facecolor,s = markersize,alpha = alpha,linewidths=lw, edgecolors=edgecolor)
    
    if correlation:
        ax.plot(x,y,c='grey')
        if show_spear:
            ax.text(0.5,0.5,f'r = {spear_r.correlation:0.3f}\n p = {spear_r.pvalue:0.3f}',transform=ax.transAxes)
            
    return ax,spear_r

# def unity_optimal_plot(axs,xdata,ydata,it,**kwargs):
#     pk = PlottingKwargs(**kwargs)
#     ax0,ax1 = axs
#     for i in range(pk.xlocs):
#         ax0.scatter(xdata[:,i], ydata[:,i])
#         ax0.plot(pk.xlocs,pk.ylocs, color = wheel.dark_grey)
#         ax0.set_xlim(pk.xlocs[0],pk.xlocs[-1])
#         ax0.set_ylim(pk.ylocs[0],pk.ylocs[-1])
#         ax0.set_xlabel('Optimal Mean Decision Time (ms)')
#         ax0.set_ylabel('Participant Mean Decision Time (ms)')
#         ax0.set_title(f'Optimal Simulation vs. Participant Mean Decision Time\nCondition: {it.trial_block_titles[i]}')


#         diff = data_metric[:,i] - all_subjects_sim_results_dict[metric][:,i] 
#         ax1.scatter(ax1_xlocs,diff)
#         ax1.axhline(zorder=0,linestyle='--')
#         max_diff = np.max(abs(diff))

#         arrow_length = max_diff 
#         head_length = 0.2*arrow_length
#         arrow_x_init = -0.25
#         arrow_y_init = max_diff/10
        
#         text_y1 = arrow_length/2 + arrow_y_init
#         text_y2 = -arrow_length/2 - arrow_y_init
        
#         ax1.arrow(arrow_x_init,arrow_y_init,0, arrow_length, width = 0.05, length_includes_head = False, head_length = head_length,head_width=0.15,shape = 'left',color=wheel.grey)
#         ax1.text(arrow_x_init/2,text_y1,'Greater Mean\nDecision Time',rotation = 90, fontweight='bold',ha='center',va='center',fontsize=9)
#         ax1.arrow(arrow_x_init,-arrow_y_init,0,-arrow_length, width = 0.05, length_includes_head = False, head_length = head_length,head_width=0.15,shape = 'right',color=wheel.grey)
#         ax1.text(arrow_x_init/2,text_y2,'Lesser Mean\nDecision Time',rotation = 90, fontweight='bold',ha='center',va='center',fontsize = 9)
        
#         ax1.set_ylim(-max_diff-(1/2)*max_diff,max_diff+(1/2)*max_diff)
#         ax1.set_xlim(-0.3,1.2)
#         ax1.set_xticks([])
#         ax1.spines.bottom.set_visible(False)
        
def multiple_models_boxplot_v2(ax,data,model_data,labels,show_boxplot=True,show_models=True,
                               **kwargs):
    pk = PlottingKwargs(**kwargs)
    w, h = plt.gcf().get_size_inches()
    if show_boxplot:
        ax,bp = multi_boxplot(ax,data,pk.xlocs,box_width = pk.bw,colors=pk.box_color,include_means=pk.include_means)
        if pk.jitter:
            dv.jitter_array(ax=ax,x_positions=pk.xlocs,data=data.T, noise_scale=0.01, include_mean = False, circle_size=30)

    if pk.line_colors is None:
        np.random.seed(1)
        pk.line_colors = dv.ColorWheel().get_random_color(n=len(model_data))

    if pk.linestyles is None:
        pk.linestyles = ['-']*len(model_data)
    
    legend_colors = []
    legend_labels = []
    legend_linestyles = []
    if show_models:
        if len(model_data)==1:
            offset = np.zeros(len(model_data))
        else:
            offset = np.linspace(-pk.bw/5,pk.bw/5,len(model_data))
        for i in range(len(model_data)):
            ax.plot(pk.xlocs+offset[i],model_data[i],c=pk.line_colors[i],marker='o',markersize=7.5, zorder=200,ls=pk.linestyles[i])
            legend_colors.append(pk.line_colors[i])
            legend_labels.append(labels[i])
            legend_linestyles.append(pk.linestyles[i])

    ax.set_xticks(pk.xtick_locs)
    ax.set_yticks(pk.ylocs)
    ax.set_xticklabels(pk.xticklabels)
    ax.set_xlabel(pk.xlabel)
    ax.set_ylabel(pk.ylabel)
    ax.set_title(pk.title, fontsize = pk.title_fontsize)
    if len(labels)>3:
        ncol = 2
    else:
        ncol=1
    dv.legend(ax,legend_labels,legend_colors,ls = legend_linestyles,loc=pk.legend_loc,
              fontsize=pk.legend_fontsize, ncol=ncol, columnspacing=1)
    
#! Legacy for Optimal_Stopping_Model_With_Data_Group
#! v2 is a much more concise implementation and is better
def multiple_models_boxplot(ax,data,show_boxplot=True,
                            known_player=None,unknown_player=None, 
                            known_optimal = None,unknown_optimal=None,
                            no_switch=None, full_fit=None,
                            **kwargs):
    '''
    unknown_player := The decision times are fit to the player, but they don't know about the gamble delay
    known_player    := The decision times are fit to the player, but they do know about the gamble delay
    unknown_optimal := The decision times are the OPTIMAL, but the optimizer does not know that there is a delay associated 
                        with switching to a guess decision
    known_optimal   := The decision times are the optimal, and the optimaizer has knowledge of the delay associated with switching to a guess decision
    '''
    bw            = kwargs.get('box_width',0.75)
    box_color     = kwargs.get('colors',wheel.seth_blue)
    xlocs         = kwargs.get('xlocs')
    legend_loc    = kwargs.get('legend_loc','best')
    include_means = kwargs.get('include_means')
    jitter        = kwargs.get('jitter',True)
    labels        = kwargs.get('labels')
    linestyles    = kwargs.get('linestyles')
    line_colors   = kwargs.get('line_colors')
    markerstyles   = kwargs.get('markerstyles')
    #* Dictionary keys
    dict_keys = ['known_player','unknown_player','known_optimal','unknown_optimal','no_switch','full_fit']
    #* Get value lists
    model_values = [known_player,unknown_player,known_optimal,unknown_optimal,no_switch, full_fit]
    label_values = [f'Model Prediction of Group\n(Account for Guessing)',
                  f'Model Prediction of Group\n(Not Account for Guessing)',
                  f'Theoretical Optimal\n(Account for Guessing)',
                  f'Theoretical Optimal\n(Not Accounting for Guessing)',
                  f'No Switch Delay',
                  f'Full Fit Model'
    ]       
    if line_colors is None:
        line_colors = [wheel.rak_red, wheel.yellow, wheel.rak_blue, wheel.light_orange, wheel.white, wheel.purple]
    if linestyles is None:
        linestyles = ['-']*len(dict_keys)
    if markerstyles is None:
        markerstyles = ['o','o','o','o','x','x']
        
    #* Create dictionaries 
    model_dict = dict(zip(dict_keys,model_values))
    labels_dict = dict(zip(dict_keys,label_values))
    line_colors_dict = dict(zip(dict_keys,line_colors))
    linestyles_dict = dict(zip(dict_keys,linestyles))
    
    #* Plot real data boxplot
    if show_boxplot:
        ax,bp = multi_boxplot(ax,data,xlocs,box_width = bw,colors=box_color,include_means=include_means)
        if jitter:
            dv.jitter_array(ax=ax,x_positions=xlocs,data=data.T, noise_scale=0.01, include_mean = False, circle_size=30)
    
    #* Plot models as lines
    legend_colors = []
    legend_labels = []
    legend_linestyles    = []
    for k in dict_keys:
        if model_dict[k] is not None:
            ax.plot(xlocs,model_dict[k],c=line_colors_dict[k],marker='o',zorder=200,ls=linestyles_dict[k])
            legend_colors.append(line_colors_dict[k])
            legend_labels.append(labels_dict[k])#\n(Gamble Delay/Uncertainty)')
            legend_linestyles.append(linestyles_dict[k])

        
    dv.legend(ax,legend_labels,legend_colors,ls = legend_linestyles,loc=legend_loc,fontsize=6)
    