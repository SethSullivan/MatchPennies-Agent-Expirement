import matplotlib.pyplot as plt
import numpy as np
import data_visualization as dv
from scipy import stats
wheel = dv.ColorWheel()

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

def unity_optimal_plot(ax,xdata,ydata,it,**kwargs):
    figsize = kwargs.get('figsize',(10,5))
    num_blocks = kwargs.get('num_blocks',6)
    for i in range(num_blocks):
        fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2.5, 1]},figsize = (10,5),dpi=125)   
        ax0.scatter(xdata[:,i], ydata[:,i])
        ax0.plot(xlocs,ylocs, color = wheel.dark_grey)
        ax0.set_xlim(xlocs[0],xlocs[-1])
        ax0.set_ylim(ylocs[0],ylocs[-1])
        ax0.set_xlabel('Optimal Mean Decision Time (ms)')
        ax0.set_ylabel('Participant Mean Decision Time (ms)')
        ax0.set_title(f'Optimal Simulation vs. Participant Mean Decision Time\nCondition: {it.trial_block_titles[i]}')


        diff = data_metric[:,i] - all_subjects_sim_results_dict[metric][:,i] 
        ax1.scatter(ax1_xlocs,diff)
        ax1.axhline(zorder=0,linestyle='--')
        max_diff = np.max(abs(diff))

        arrow_length = max_diff 
        head_length = 0.2*arrow_length
        arrow_x_init = -0.25
        arrow_y_init = max_diff/10
        
        text_y1 = arrow_length/2 + arrow_y_init
        text_y2 = -arrow_length/2 - arrow_y_init
        
        ax1.arrow(arrow_x_init,arrow_y_init,0, arrow_length, width = 0.05, length_includes_head = False, head_length = head_length,head_width=0.15,shape = 'left',color=wheel.grey)
        ax1.text(arrow_x_init/2,text_y1,'Greater Mean\nDecision Time',rotation = 90, fontweight='bold',ha='center',va='center',fontsize=9)
        ax1.arrow(arrow_x_init,-arrow_y_init,0,-arrow_length, width = 0.05, length_includes_head = False, head_length = head_length,head_width=0.15,shape = 'right',color=wheel.grey)
        ax1.text(arrow_x_init/2,text_y2,'Lesser Mean\nDecision Time',rotation = 90, fontweight='bold',ha='center',va='center',fontsize = 9)
        
        ax1.set_ylim(-max_diff-(1/2)*max_diff,max_diff+(1/2)*max_diff)
        ax1.set_xlim(-0.3,1.2)
        ax1.set_xticks([])
        ax1.spines.bottom.set_visible(False)

def multiple_models_boxplot_v2(ax,data,model_data_list,labels,show_boxplot=True,
                            **kwargs):
    bw            = kwargs.get('box_width',0.75)
    box_color     = kwargs.get('colors',wheel.seth_blue)
    xlocs         = kwargs.get('xlocs')
    legend_loc    = kwargs.get('legend_loc','best')
    include_means = kwargs.get('include_means')
    jitter        = kwargs.get('jitter',True)
    remove_parentheses_from_labels = kwargs.get('remove_parentheses_from_labels',False)
    linestyles = kwargs.get('linestyles')
    if show_boxplot:
        ax,bp = multi_boxplot(ax,data,xlocs,box_width = bw,colors=box_color,include_means=include_means)
        if jitter:
            dv.jitter_array(ax=ax,x_positions=xlocs,data_list=data.T, noise_scale=0.01, include_mean = False, circle_size=30)
    
    line_colors = kwargs.get('line_colors')
    if line_colors is None:
        line_colors = [wheel.rak_red,wheel.yellow,wheel.rak_blue,wheel.light_orange]
        
    if linestyles is None:
        linestyles = ['-','-','-','-']
    legend_colors = []
    legend_labels = []
    legend_linestyles    = []
    
    for i in range(len(model_data_list)):
        ax.plot(xlocs,model_data_list[i],c=line_colors[i],marker='*',markersize=10, zorder=200,ls=linestyles[i])
        legend_colors.append(line_colors[i])
        legend_labels.append(labels[i])
        legend_linestyles.append(linestyles[i])
        
    dv.legend(ax,legend_labels,legend_colors,ls = legend_linestyles,loc=legend_loc,fontsize=9)

def multiple_models_boxplot(ax,data,show_boxplot=True,
                            known_player=None,unknown_player=None, 
                            known_optimal = None,unknown_optimal=None,
                            no_switch=None,
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
    linestyles = kwargs.get('linestyles')
    line_colors = kwargs.get('line_colors')
    markerstyles   = kwargs.get('markerstyles')
    #* Dictionary keys
    dict_keys = ['known_player','unknown_player','known_optimal','unknown_optimal','no_switch']
    #* Get value lists
    model_values = [known_player,unknown_player,known_optimal,unknown_optimal,no_switch]
    label_values = [f'Model Prediction of Group\n(Account for Guessing)',
                  f'Model Prediction of Group\n(Not Account for Guessing)',
                  f'Theoretical Optimal\n(Account for Guessing)',
                  f'Theoretical Optimal\n(Not Accounting for Guessing)',
                  f'No Switch Delay'
    ]       
    if line_colors is None:
        line_colors = [wheel.rak_red,wheel.yellow,wheel.rak_blue,wheel.light_orange, wheel.white]
    if linestyles is None:
        linestyles = ['-']*len(dict_keys)
    if markerstyles is None:
        markerstyles = ['o','o','o','o','x']
        
    #* Create dictionaries 
    model_dict = dict(zip(dict_keys,model_values))
    labels_dict = dict(zip(dict_keys,label_values))
    line_colors_dict = dict(zip(dict_keys,line_colors))
    linestyles_dict = dict(zip(dict_keys,linestyles))
    
    #* Plot real data boxplot
    if show_boxplot:
        ax,bp = multi_boxplot(ax,data,xlocs,box_width = bw,colors=box_color,include_means=include_means)
        if jitter:
            dv.jitter_array(ax=ax,x_positions=xlocs,data_list=data.T, noise_scale=0.01, include_mean = False, circle_size=30)
    
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
    