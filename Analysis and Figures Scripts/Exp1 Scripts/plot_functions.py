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
