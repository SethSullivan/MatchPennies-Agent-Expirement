import matplotlib.pyplot as plt
import numpy as np
import data_visualization as dv
from scipy import stats
from copy import deepcopy
import matplotlib as mpl
import matplotlib.transforms as mtransforms
from string import ascii_uppercase
from functools import partial
wheel = dv.ColorWheel()
class NewFigure:
    def __init__(self, mosaic, figsize = (6.5,4), dpi=150, layout="constrained", sharex=False,sharey=False,
                 hspace = None, wspace = None, height_ratios=None, width_ratios=None):
        
        self.fig, self.axes = plt.subplot_mosaic(mosaic, 
                                                 dpi=dpi,
                                                layout=layout,
                                                figsize=figsize,
                                                sharex=sharex,
                                                sharey=sharey,
                                                gridspec_kw={'wspace':wspace,"hspace":hspace},
                                                height_ratios=height_ratios,
                                                width_ratios=width_ratios,
                                                )
        self.num_axes = len(self.axes.values())
        self.figw,self.figh = self.fig.get_size_inches()
        # Create axmain box for visualization of the bounds and coordinates
        self.axmain  = self.fig.add_axes((0,0,1,1))
        self.axmain.set_xlim(0,self.figw)
        self.axmain.set_ylim(0,self.figh)
        for spine in ['top','right','bottom','left']:
            self.axmain.spines[spine].set_visible(True)
        self.axmain.set_xlim(0,figsize[0])
        self.axmain.set_ylim(0,figsize[1])
        
        self.letters = []  
        
    def fig_data_to_axis_transform(self,ax):
        '''
        Transformation for figure data coordinates (aka axmain) to ax coordinates desired
        '''
        return self.axmain.transData + ax.transAxes.inverted()
    
    def axis_to_fig_data_transform(self,ax):
        return ax.transAxes + self.axmain.transData.inverted()
                
    def pad_fig(self, w_pad, h_pad, w_space, h_space):
        self.fig.get_layout_engine().set(w_pad=w_pad/self.figw, 
                                         h_pad=h_pad/self.figh, 
                                         wspace=w_space/self.figw,
                                         hspace=h_space/self.figh)
    def ax_dim(self,ax):
        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        return bbox.width, bbox.height
    
    def ax_loc(self,ax):
        bbox = ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        return bbox.x0,bbox.y0
        
    def set_position(self, ax, loc):
        self.fig.canvas.draw()
        x,y = loc
        w,h = self.ax_dim(ax)
        self.fig.set_layout_engine('none')

        ax.set_position((x/self.figw, y/self.figh, w/self.figw, h/self.figh))

    def adjust_position(self, ax, adjustment):
        self.fig.canvas.draw()

        dx,dy = adjustment
        w,h = self.ax_dim(ax)
        x,y = self.ax_loc(ax)
        self.fig.set_layout_engine('none')
        ax.set_position(((x+dx)/self.figw, (y+dy)/self.figh, w/self.figw, h/self.figh))
        
    def set_size(self, ax, size):
        self.fig.canvas.draw()
        w,h = size
        x,y = self.ax_loc(ax)
        self.fig.set_layout_engine('none')
        ax.set_position((x/self.figw, y/self.figh, w/self.figw, h/self.figh))
        
    def adjust_size(self, ax, adjustment):
        self.fig.canvas.draw()
        dw,dh = adjustment
        w,h = self.ax_dim(ax)
        x,y = self.ax_loc(ax)
        self.fig.set_layout_engine('none')
        ax.set_position((x/self.figw, y/self.figh, (w+dw)/self.figw, (h+dh)/self.figh))

    # def add_all_letters_v2(self, xy = (0,1), ax=None, fontsize=12,
    #                     va="top",ha='left',fontfamily="sans-serif",fontweight="bold",
    #                     verticalshift=0, horizontalshift=0):
    #     if not isinstance(verticalshift, list):
    #         verticalshift = np.array([verticalshift]*self.num_axes)
    #     if not isinstance(horizontalshift, list):
    #         horizontalshift = np.array([horizontalshift]*self.num_axes)
            
    #     for i,(label, ax) in enumerate(self.axes.items()):
    #         # label physical distance in and down:
    #         ax.get_tightbbox()
    #         letter = ascii_uppercase[i]
    #         trans = self.fig_data_to_axis_transform(ax)
    #         ax.text(-0.15 + horizontalshift[i], 1.1 + verticalshift[i], letter, transform=ax.transAxes,
    #                 fontsize=fontsize, verticalalignment=va, ha=ha,
    #                 fontfamily=fontfamily, fontweight="bold",
    #         )
            
    def add_all_letters(self, xy = (0,1), fontsize=12,
                        va="top",ha='left',fontfamily="sans-serif",fontweight="bold",
                        verticalshift=0, horizontalshift=0):
        default_start = (-0,1.0)
        if not isinstance(verticalshift, list):
            verticalshift = np.array([verticalshift]*self.num_axes)
        if not isinstance(horizontalshift, list):
            horizontalshift = np.array([horizontalshift]*self.num_axes)
        
        shift = np.vstack((horizontalshift, verticalshift))
        
        for i,(label, ax) in enumerate(self.axes.items()):
            transfig_loc = self.axis_to_fig_data_transform(ax).transform(default_start) + shift[:,i]
            transax_loc = self.fig_data_to_axis_transform(ax).transform(transfig_loc)
            # label physical distance in and down:
            letter = ascii_uppercase[i]
            trans = self.fig_data_to_axis_transform(ax)
            ax.text(transax_loc[0], transax_loc[1], letter, transform=ax.transAxes,
                    fontsize=fontsize, verticalalignment=va, ha=ha,
                    fontfamily=fontfamily, fontweight="bold",
            )
                   
            
    def add_letter(self, ax, x, y, letter = None, fontsize = 12, 
                   ha = "left", va = "top", color = "black", zorder = 20, transform = None):
        if letter == None:
            letter_to_add = ascii_uppercase[len(self.letters)]
        else:
            letter_to_add = letter
        if transform is None:
            transform = ax.transAxes
        
        self.letters.append(letter_to_add)
        ax.text(x, y, letter_to_add, ha = ha, va = va, transform=transform,
                fontweight = "bold", color = color, fontsize = fontsize, zorder = zorder)
    
    @property
    def alphabetic_axes(self) -> list:
        sorted_keys = sorted(self.axes)
        return {k:self.axes[k] for k in sorted_keys}
        
        
        
    def remove_figure_borders(self):
        # for spine in ['top','right','bottom','left']:
        self.axmain.axis("off")
        
    def savefig(self,path,dpi=300, transparent = True):
        self.remove_figure_borders()
        self.fig.savefig(path,dpi=dpi,transparent=transparent)
class PrettyTable:
    def __init__(self, ax, table_values: np.ndarray,
                 text_xshift=0.5, line_yshift=0, ha='center',va='center', 
                 fontsize=9, fontweight='light', fontcolor=wheel.grey,
                 inner_horizontal_ls='-', inner_vertical_ls='-', 
                 inner_vertical_lw=1.0,inner_horizontal_lw=1.0, 
                 inner_line_color = 'grey', border_color='grey',
                 border_fill=None, border_lw=1, 
                 border_ls='-', bold_first_row=False, 
                 bold_first_column=False):
        self.table_values = table_values
        
        self.ha = self._check_kwargs(ha, "ha", str)
        self.va = self._check_kwargs(va, "va", str)
        fontweight = self._check_kwargs(fontweight, "fontweight", str)
        fontsize = self._check_kwargs(fontsize, "fontsize", (float, int))
        fontcolor = self._check_kwargs(fontcolor, "fontcolor", str)
        inner_horizontal_ls = self._check_kwargs(inner_horizontal_ls, "inner_horizontal_ls", str, 
                                                 fill_function=partial(np.full_like, 
                                                                                 a=self.table_values[:,0])
                                                 )
        inner_vertical_ls = self._check_kwargs(inner_vertical_ls, "inner_vertical_ls", str, 
                                                 fill_function=partial(np.full_like, 
                                                                                 a=self.table_values[0,:])
                                                 )
        
        inner_horizontal_lw = self._check_kwargs(inner_horizontal_lw, "inner_horizontal_lw", float, 
                                                 fill_function=partial(np.full_like, 
                                                                        a=self.table_values[:,0])
                                                 )
        inner_vertical_lw = self._check_kwargs(inner_vertical_lw, "inner_vertical_lw", float, 
                                                 fill_function=partial(np.full_like, 
                                                                        a=self.table_values[0,:])
                                                 )
        
        self.ax = ax
        self.num_rows, self.num_cols = table_values.shape 
        
        self.ax.set_alpha(0)
        self.ax.patch.set_alpha(0)
        self.ax.set_xlim(0,self.num_cols)
        self.ax.set_ylim(0,self.num_rows)
        self.ax.invert_yaxis()
        
        if bold_first_row:
            fontweight[0,:] = "bold"
        if bold_first_column:
            fontweight[:,0] = "bold"
            
        self._plot_table_values(text_xshift, fontsize, fontweight, fontcolor=fontcolor)
        self._plot_table_lines(line_yshift, inner_horizontal_ls=inner_horizontal_ls,
                               inner_vertical_ls=inner_vertical_ls,
                               inner_line_color=inner_line_color, 
                               inner_horizontal_lw=inner_horizontal_lw, 
                               inner_vertical_lw=inner_vertical_lw,)
        self._plot_table_boundary(line_yshift, border_color=border_color, 
                                  border_fill=border_fill, border_lw=border_lw)
    
    def _check_kwargs(self, kwarg, kwarg_name, dtype, fill_function=None):
        '''
        Checks if it's the right type and if it's not an array, creates an array
        '''
        
        if fill_function is None:
            fill_function = partial(np.full_like, a=self.table_values)
            
        if isinstance(kwarg,dtype):
            kwarg = fill_function(fill_value=kwarg, dtype=object)
        elif isinstance(kwarg,(list, np.ndarray)):
            kwarg = kwarg
        else:
            raise ValueError(f"'{kwarg_name}' should be a {dtype} or an array of {dtype} with the same shape as 'table_values'")
        return kwarg
    
    def _plot_table_values(self, text_xshift, fontsize, fontweight, fontcolor):
        self.coordinate_store = []
        for i,row in enumerate(self.table_values):
            for j, element in enumerate(row):
                t = self.ax.text(x=j+text_xshift, y=i+0.5, s=element, ha=self.ha[i,j], va=self.va[i,j], transform=self.ax.transData, 
                        fontsize=fontsize[i,j],fontweight=fontweight[i,j], color=fontcolor[i,j])        
                self.coordinate_store.append((i,j, i+(1/self.num_cols), j+(1/self.num_rows)))
    
    def _plot_table_lines(self, line_yshift, inner_horizontal_ls,
                          inner_vertical_ls, inner_line_color, 
                          inner_horizontal_lw, inner_vertical_lw):
        for i in range(1, self.table_values.shape[0]):
            self.ax.plot([0,self.num_cols],
                         [i+line_yshift, i+line_yshift],
                         ls=inner_horizontal_ls[i-1],
                         lw=inner_horizontal_lw[i-1],
                         c=inner_line_color,
                         transform=self.ax.transData, 
                         clip_on=False)
            
        for j in range(1,self.table_values.shape[1]):
            self.ax.plot([j,j],
                         [0+line_yshift, self.num_rows+line_yshift],
                         ls=inner_vertical_ls[j-1],
                         lw=inner_vertical_lw[j-1],
                         c=inner_line_color,
                         transform=self.ax.transData,
                         clip_on=False)
            
    def _plot_table_boundary(self,line_yshift, border_color, border_fill, border_lw):
        rect = mpl.patches.Rectangle((0,0),width = self.num_cols, height=self.num_rows+abs(line_yshift), 
                                     fill=border_fill, edgecolor=border_color,lw=border_lw, clip_on=False)
        self.ax.add_patch(rect)
                
    def fill_cells(self, cell_indices, facecolor, 
                   alpha=0.5, edgecolor=None):
        '''
        t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))
        '''
        if not isinstance(cell_indices,list):
            cell_indices = [cell_indices]
            
        for cell_index in cell_indices:
            self.text_store[cell_index].set_bbox(dict(facecolor=facecolor, edgecolor=edgecolor, alpha=alpha))            
class PlottingKwargs:
    def __init__(self,**kwargs):
        self.bw            = kwargs.get('box_width',0.75)
        self.one_box_color = kwargs.get('one_box_color',wheel.seth_blue)
        self.box_colors    = kwargs.get('box_colors', None)
        self.jitter_data   = kwargs.get('jitter_data',False)
        self.jitter_color  = kwargs.get('jitter_color',None)
        self.xlocs         = kwargs.get('xlocs')
        self.xtick_locs    = kwargs.get('xtick_locs', deepcopy(self.xlocs))
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
        self.title_padding = kwargs.get('title_padding',0)
        self.legend_fontsize = kwargs.get('legend_fontsize',6)
        self.legend_linewidth = kwargs.get('legend_linewidth',3)
        self.title_fontsize = kwargs.get('title_fontsize',20)
        self.box_lw       = kwargs.get("box_lw", 2.)
        self.box_width    = kwargs.get("box_width", .75)
        self.whisker_lw   = kwargs.get("whisker_lw", 2.)
        self.reorder_xaxis = kwargs.get('reorder_xaxis',False)
        self.model_markersize = kwargs.get('model_markersize',8)
        self.data_markersize  = kwargs.get('data_markersize',50)
        self.axeslabel_fontsize = kwargs.get('axeslabel_fontsize',14)
        self.ticklabel_fontsize = kwargs.get('ticklabel_fontsize',13)

def make_figure_panel(figsize, inset_size, dpi = 100):
    fig,axmain = plt.subplots(dpi = dpi, figsize = figsize)
    
    axmain.set_xlim(0,figsize[0])
    axmain.set_ylim(0,figsize[1])
    axmain.spines.right.set_visible(True)
    axmain.spines.top.set_visible(True)
    ax = axmain.inset_axes(inset_size,transform=axmain.transData)
    
    return axmain,ax
def boxplot(ax, x, data, jitter_data = False, clip_on = True, **kwargs):

    """
    A function to plot a box plot on a given matplotlib axis with additional options for jittered data. Will omit nans from the data. 
    Parameters:
    ax : matplotlib axis object
        The axis object to plot the box plot on.
    x : float or int
        The x position of the box plot on the axis.
    data : array-like (1D)
        The data to plot in the box plot.
    jitter_data : bool, optional
        If True, adds jittered data to the plot. Default is False.
    **kwargs : keyword arguments, optional
    
    Additional options to customize the plot, including:
        - linewidth (float): The width of the box plot and whiskers. Default is 1.2.
        - box_lw (float): The width of the box edges. Default is the linewidth value.
        - box_width (float): The width of the box. Default is 0.5.
        - whisker_lw (float): The width of the whiskers. Default is the box_lw value.
        - color (str): The color of the box plot and whiskers. Default is "#0BB8FD".
        - noise_scale (float): The scale of the jittered noise added to the data points. Default is 0.06.
        - data_color (str): The color of the jittered data points. Default is the color value.
        - include_mean (bool): If True, adds a marker for the mean of the data. Default is False.

        - data_lw (float): The width of the jittered data points. Default is 0.5.
        - data_size (int): The size of the jittered data points. Default is 40.
        - data_alpha (float): The alpha value of the jittered data points. Default is 1.
        - data_zorder (int): The z-order of the jittered data points. Default is 0.

        - mean_size (int): The size of the mean marker. Default is the data_size value.
        - mean_alpha (float): The alpha value of the mean marker. Default is 1.
        - mean_color (str): The color of the mean marker. Default is "#727273".
        - mean_zorder (int): The z-order of the mean marker. Default is 0.

    Returns:
    ax : matplotlib axis object
        The axis object with the box plot and optional jittered data added.
    """
    if "lw" in kwargs.keys() and "linewidth" in kwargs.keys():
        raise ValueError("Keyword argument repeated.")
        
    linewidth = kwargs.get("linewidth", 1.2)
    lw = kwargs.get("lw", linewidth)
    
    box_lw       = kwargs.get("box_lw", lw)
    box_width    = kwargs.get("box_width", .5)
    whisker_lw   = kwargs.get("whisker_lw", box_lw)

    color = kwargs.get("color",   "#0BB8FD")

    #box properties
    box_props   = {"facecolor": "none", "edgecolor" : color, "linewidth": box_lw, "alpha": 1}

    #whisker properties
    whisker_props = {"linewidth" : whisker_lw, "color": color}

    #cap properties
    cap_props = {"linewidth" : whisker_lw, "color": color}

    #median properties
    median_props = {"linewidth" : whisker_lw, "color": color}

    '''Make Box Plots'''
    ax.set_facecolor("none")
    
    data = np.array(data)
    filtered_data = data[~np.isnan(data)]
    #Make Box
    bp = ax.boxplot([filtered_data],   positions = [x], patch_artist = True,  showfliers = False, 
                boxprops = box_props  , whiskerprops = whisker_props,
                capprops = cap_props, medianprops = median_props, widths = box_width)
    
    #Add jittered data
    if jitter_data:
        noise_scale = kwargs.get("noise_scale", .06)
        data_color = kwargs.get("data_color", color)
        include_mean = kwargs.get("include_mean", False)
        
        data_lw = kwargs.get("data_lw", .5)
        data_size = kwargs.get("data_size", 40)
        data_alpha = kwargs.get("data_alpha", 1)
        data_zorder = kwargs.get("data_zorder", 0)
        
        mean_size = kwargs.get("mean_size", data_size)
        mean_alpha = kwargs.get("mean_alpha", 1)
        mean_color = kwargs.get("mean_color", '#727273')
        mean_zorder = kwargs.get("mean_zorder", 0)

        noise = np.random.normal(0, noise_scale, len(filtered_data))
        
        ax.scatter(x + noise, filtered_data,
                s = data_size, facecolors = 'none',
               edgecolors=data_color, alpha = data_alpha, lw = data_lw, zorder = data_zorder, clip_on = clip_on)
            
        if include_mean:
            ax.scatter(x, np.nanmean(filtered_data) ,
                        s = data_size, facecolors = mean_color,
                       edgecolors=mean_color, alpha = mean_alpha, lw = data_lw, zorder = mean_zorder, clip_on = clip_on)
        
    return ax

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
    colors       = kwargs.get("colors", None)
    one_color    = kwargs.get('one_color',wheel.grey)
    #box properties
    box_props,whisker_props,cap_props,median_props = {},{},{},{}
    box_props = {"facecolor": "none", "edgecolor" : one_color, "linewidth": box_lw, "alpha": 1}

    #whisker properties
    whisker_props = {"linewidth" : whisker_lw, "color": one_color}

    #cap properties
    cap_props = {"linewidth" : whisker_lw, "color": one_color}

    #median properties
    median_props = {"linewidth" : whisker_lw, "color": one_color}

    include_means = kwargs.get('include_means')
    '''Make Box Plots'''
    ax.patch.set_alpha(0)
    
    if np.isnan(data).any():
        mask = ~np.isnan(data)
        filtered_data = [d[m] for d,m in zip(data.T, mask.T)]
    else:
        filtered_data = data
    
    bp = ax.boxplot(filtered_data,   positions = xlocs, patch_artist = True,  showfliers = False, 
                boxprops = box_props , 
                whiskerprops = whisker_props,
                capprops = cap_props, 
                medianprops = median_props, 
                widths = box_width,
                )
    if colors is not None:
        for element in bp:
            for patch, color in zip(bp[element],colors):
                print(patch)
                patch.set_color(color)
        
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

def multiple_models_boxplot(ax,data,model_data,labels,
                            show_boxplot=True,show_models=True,
                            **kwargs):
    pk = PlottingKwargs(**kwargs)
    model_data = np.array(model_data)
    w, h = plt.gcf().get_size_inches()
    if pk.reorder_xaxis:
        pk.xticklabels = pk.xticklabels[::2] + pk.xticklabels[1::2]
        data = data[:,[0,2,4,1,3,5]]
        if model_data.ndim == 1:
            model_data = np.array(model_data)[[0,2,4,1,3,5]]
        elif model_data.ndim == 2:
            model_data = np.array(model_data)[:,[0,2,4,1,3,5]]

    if show_boxplot:
        # ax,bp = multi_boxplot(ax,data,pk.xlocs,box_width = pk.bw,
        #                       one_color=pk.one_box_color,colors=pk.box_colors, 
        #                       include_means=pk.include_means)
        for i in range(len(pk.xlocs)):
            dv.boxplot(ax,pk.xlocs[i],data=data[:,i],color = pk.box_colors[i],data_color=pk.box_colors[i], **pk.__dict__)
        
        if pk.jitter:
            if pk.jitter_color is None:
                pk.jitter_color = pk.box_colors
            np.random.seed(0)
            dv.jitter_array(ax=ax,x_positions=pk.xlocs,data=data.T, data_color = pk.jitter_color, data_edge_color = wheel.lighten_color(wheel.light_grey,1.2), 
                            noise_scale=0.06, include_mean = False, circle_size=pk.data_markersize)

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
            offset = np.linspace(-pk.bw/5,pk.bw/5,len(model_data))[::-1]
        for i in range(len(model_data)):
            ax.plot(pk.xlocs+offset[i],model_data[i],c=pk.line_colors[i],marker='o',
                    markersize=pk.model_markersize, zorder=200,ls=pk.linestyles[i], clip_on=False)
            legend_colors.append(pk.line_colors[i])
            legend_labels.append(labels[i])
            legend_linestyles.append(pk.linestyles[i])

    ax.set_xticks(pk.xtick_locs)
    ax.set_yticks(pk.ylocs)
    ax.set_xticklabels(pk.xticklabels,fontsize=pk.ticklabel_fontsize,fontweight='bold')
    [ticklabel.set_color(color) for (color,ticklabel) in zip(pk.box_colors,ax.xaxis.get_ticklabels())]
    ax.set_yticklabels(pk.ylocs,fontsize=pk.ticklabel_fontsize)
    ax.set_xlabel(pk.xlabel,labelpad=4,fontsize=pk.axeslabel_fontsize)
    ax.set_ylabel(pk.ylabel,labelpad=1,fontsize=pk.axeslabel_fontsize)
    ax.set_title(pk.title, fontsize = pk.title_fontsize, pad=pk.title_padding)
    if len(labels)>3:
        ncol = 2
    else:
        ncol=1
    dv.legend(ax,legend_labels,legend_colors,ls = legend_linestyles,loc=pk.legend_loc,
              fontsize=pk.legend_fontsize, ncol=ncol, columnspacing=1,linewidth=pk.legend_linewidth)
    
def plot_stats_v2(ax, pvals:list[dict],cles, combos:list[str], 
               ypos:list[int], xlocs:list[list[float,float]], 
               show_effectsize=False, **kwargs):
    '''
    stacked: 
    fontsize: 
    h: 
    lw: 
    '''
        
    for i in range(len(combos)):
        if show_effectsize:
            kwargs.update({'cles':cles.iloc[0][combos[i]]})
        else:
            kwargs.update({'cles':None})
        pval = pvals.iloc[0][combos[i]]
        if pval>1.0:
            pval = 1.0
        dv.stat_annotation(ax,xlocs[i][0],xlocs[i][1],y=ypos[i],
                           p_val= pval, **kwargs)

def plot_stats(ax, statistics:list[dict], combos:list[str], 
               ypos:list[int], xpositions:list[list[float,float]], 
               show_effectsize=False, **kwargs):
    '''
    stacked: 
    fontsize: 
    h: 
    lw: 
    '''
        
    for i in range(len(combos)):
        if show_effectsize:
            kwargs.update({'cles':statistics[1][combos[i]]})
        else:
            kwargs.update({'cles':None})
        dv.stat_annotation(ax,xpositions[i][0],xpositions[i][1],y=ypos[i],
                           p_val= statistics[0][combos[i]], **kwargs)

def plot_boostrapped_model_results(ax, x, y, percentiles, color, horizontal_lw = 0.2, markersize=10):
    left = x - horizontal_lw/2
    right = x + horizontal_lw/2
    if percentiles is not None:
        top = percentiles[0]
        bottom = percentiles[1]
        ax.plot([x, x], [top, bottom], color=color)
        ax.plot([left, right], [top, top], color=color)
        ax.plot([left, right], [bottom, bottom], color=color)
    ax.plot(x, y, 'o', color=color, markersize=markersize, zorder=99, clip_on=False)
    
def plot_models(ax, xlocs, data, line_colors, linestyles, bw, 
                markersize, legend_labels, legend_linewidth, 
                legend_loc="best", legend_fontsize=10, 
                ncol=1, columnspacing=1, legend=True):
    '''
    data can have multiple models in it
    '''
    legend_colors = []
    legend_linestyles = []
    if len(data)==1:
        offset = np.zeros(len(data))
    else:
        offset = np.linspace(-bw/5,bw/5,len(data))[::-1]
    for i in range(len(data)):
        ax.plot(xlocs+offset[i], data[i], c=line_colors[i], marker='o',
                markersize=markersize, zorder=200, ls=linestyles[i], clip_on=False)
        legend_colors.append(line_colors[i])
        legend_linestyles.append(linestyles[i])
    if legend:
        dv.legend(ax,legend_labels,legend_colors,ls = legend_linestyles,loc=legend_loc,
                fontsize=legend_fontsize, ncol=ncol, columnspacing=columnspacing, linewidth=legend_linewidth)

def exp2_reaction_plots(ax, metric, pvals, cles, 
                        ylocs, statline_y, shift_statlines, 
                        h, colors =[wheel.rak_red, wheel.rak_blue, wheel.dark_red, wheel.dark_blue]
    ):
    
    for i in range(4):
        dv.boxplot(metric[:,i],x_pos = i,ax=ax,box_lw=3, linewidth=3,whisker_lw=3, color = colors[i])
    dv.jitter_array(ax =ax, x_positions = xlocs,
                    data = metric.T,
                    circle_size = 75,include_mean=False,
                    data_color = colors, data_edge_color = wheel.light_grey, circle_lw=1.)
    np.random.seed(2)
    xlocs = np.arange(0,5,1)
    combos = ['01','02','13','23',]
    for i,c in enumerate(combos):
        a = int(c[0])
        b = int(c[1])
        if pvals[c]<0.05:
            dv.stat_annotation(ax,xlocs[a],xlocs[b],statline_y,p_val=pvals[c],cles=cles[c],
                            fontsize=12,h=h, lw=1.25)
            statline_y += shift_statlines[i]
    

   
#! Legacy for Optimal_Stopping_Model_With_Data_Group
#! v2 is a much more concise implementation and is better
def multiple_models_boxplot_old(ax,data,show_boxplot=True,
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
    