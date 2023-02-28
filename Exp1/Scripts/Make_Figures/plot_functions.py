import matplotlib.pyplot as plt
import numpy as np
import data_visualization as dv
from scipy import stats
wheel = dv.ColorWheel()


#%%
def make_figure_panel(figsize, inset_size, dpi = 125):
    fig,axmain = plt.subplots(dpi = dpi, figsize = figsize)
    
    axmain.set_xlim(0,figsize[0])
    axmain.set_ylim(0,figsize[1])
    axmain.spines.right.set_visible(True)
    axmain.spines.top.set_visible(True)
    ax = axmain.inset_axes(inset_size,transform=axmain.transData)
    
    return axmain,ax
#%% Box plot with option to make double boxplot
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
    box_width    = kwargs.get("box_width", .5)
    whisker_lw   = kwargs.get("whisker_lw", 2.0)

    color = kwargs.get("colors",  "#0BB8FD")
    #box properties
    box_props,whisker_props,cap_props,median_props = {},{},{},{}
    box_props = {"facecolor": "none", "edgecolor" : color, "linewidth": box_lw, "alpha": 1}

    #whisker properties
    whisker_props = {"linewidth" : whisker_lw, "color": color}

    #cap properties
    cap_props = {"linewidth" : whisker_lw, "color": color}

    #median properties
    median_props = {"linewidth" : whisker_lw, "color": color}


    '''Make Box Plots'''
    ax.patch.set_alpha(0)
    
    if np.isnan(data).any():
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d,m in zip(data.T, mask.T)]
    else:
        filtered_data = data
    
    bp = ax.boxplot(filtered_data,   positions = xlocs, patch_artist = True,  showfliers = False, 
                boxprops = box_props  , whiskerprops = whisker_props,
                capprops = cap_props, medianprops = median_props, widths = box_width)
    
    # for element in bp:
    #     for patch, color in zip(bp[element],colors):
    #         print(patch)
    #         patch.set_color(color)
        
    return ax
    
#%%
def scatter_with_correlation(ax,xdata,ydata,**kwargs):
    facecolor = kwargs.get('facecolor','none')
    edgecolor = kwargs.get('edgecolor',wheel.seth_blue)
    markersize = kwargs.get('markersize',30)
    alpha = kwargs.get('alpha',1.0)
    lw = kwargs.get('linewidths',1.0)
    correlation = kwargs.get('correlation',True)
    
    xdata = xdata.flatten()
    ydata= ydata.flatten()
    lm = stats.linregress(xdata,ydata)
    x = np.arange(np.nanmin(xdata)-10,np.nanmax(xdata)+10,1)
    y = lm.slope*x + lm.intercept
    spear_r = stats.spearmanr(xdata, ydata)
    ax.scatter(xdata,ydata,c=facecolor,s = markersize,alpha = alpha,linewidths=lw, edgecolors=edgecolor)
    if correlation:
        ax.plot(x,y,c='grey')

    return ax,spear_r

#%%
def unity_optimal_plot(ax,xdata,ydata,**kwargs):
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
        ax0.set_title(f'Optimal Simulation vs. Participant Mean Decision Time\nCondition: {trial_block_titles[i]}')


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