''' 
Author: Lucas Rath
'''
import numpy as np
from scipy import stats
from typing import List
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import pandas as pd
import plotly
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from CBOSS.models.models import Sparse_Identification_Nonlinear_Dynamical_Systems
try:
    import aim
except:
    pass


def gen_configuration_graph_2D(X:np.ndarray):
        # find Adjacency Matrix of the Hamming Graph
    X1, X2 = [X.astype(int)] * 2
    diff = X1[:,None] - X2[None,:]
    diff[np.abs(diff) > 1e-5] = 1
    hamming_dist = diff.sum(axis=-1)
    adj_matrix = hamming_dist.copy()
    adj_matrix[adj_matrix > 1] = 0

    G = nx.from_pandas_adjacency(df=pd.DataFrame(data=adj_matrix))

    layout = nx.spring_layout(G, weight=0.01, k=0.2)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    node_x = []
    node_y = []
    for node in G.nodes():
        _x, _y = layout[node]
        node_x.append(_x)
        node_y.append(_y)
    
    return np.asarray(node_x), np.asarray(node_y), np.asarray(edge_x), np.asarray(edge_y)

def print_opt_progress(log:dict, evalBudget:int, *args, **kwargs):
    with np.printoptions(edgeitems=50, linewidth=100000):
        print(f"nevals: {len(log['y'])}/{evalBudget}  time: {log['walltime'][-1]/60.:.1f}min  y_feas_min: {np.nanmin(log['y'][(log['c']<=0).all(1)]):.3f}")

def aim_plot_opt_progress(log:dict, aim_run:'aim.Run'=None, *args, **kwargs):
    
    X, y, c, l = log.get('X', None), log.get('y', None), log.get('c', None), log.get('l', None)
    walltime = log.get('walltime', None)
    y_pred_mean, y_pred_std = log.get('y_pred_mean', None), log.get('y_pred_std', None)
    c_pred_mean, c_pred_std = log.get('c_pred_mean', None), log.get('c_pred_std', None)
    l_pred_prob = log.get('l_pred_prob', None)
    
    # number of constraints
    nc = c.shape[1] if c is not None else 0
    
    fig = make_subplots(rows = 1 + nc + 1, cols=1, shared_xaxes=True)      # 1 plot for the cost function, nc plots for the constraint, and 1 plot for l
    
    ''' axis1 - Cost Function 
    ----------------------------'''
    idx_feas = np.where((c <= 0).all(1))[0]
    if y_pred_mean is not None:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y))), y=y_pred_mean[:,0],
                marker=dict(color='rgba(1,1,1,0.2)', size=6, opacity=0.5),
                name="Predictions",
                mode='markers',
                error_y=dict(type='data', visible=True, array=y_pred_std, color='rgba(1,1,1,0.2)'),
                legendgroup = '1'
            ),
            row=1, col=1
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(y))), y=y[:,0],
            marker=dict(color='red', size=6, opacity=0.5),
            name="Unfeasible Evaluations",
            mode='markers',
            legendgroup = '1'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=idx_feas, y=np.fmin.accumulate(y[idx_feas,0]),
            line=dict(color='blue'),
            line_shape='hv',
            name="Best feasible",
            legendgroup = '1',
            mode='lines'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=idx_feas, y=y[idx_feas,0],
            marker=dict(color='blue', size=6),
            name="Feasible Evaluations",
            mode='markers',
            legendgroup = '1'
        ),
        row=1, col=1
    )

    ''' axis2 - Constraints 
    ----------------------------'''
    for ic in range(nc):
        # ic = 0
        fig.add_trace(
            go.Scatter(
                x=np.where(c[:,ic]>0)[0], y=c[c[:,ic]>0,ic],
                marker=dict(color='red', size=6), # , opacity=0.5
                name='Unfeasible Constraint Evaluations',
                mode='markers',
                legendgroup = '2'
            ),
            row=1+nc, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=np.where(c[:,ic]<=0)[0], y=c[c[:,ic]<=0,ic],
                marker=dict(color='blue', size=6), # , opacity=0.5
                name='Feasible Constraint Evaluations',
                mode='markers',
                legendgroup = '2'
            ),
            row=1+nc, col=1
        )
        if c_pred_mean is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(c))), y=c_pred_mean[:,ic],
                    marker=dict(color='rgba(1,1,1,0.2)', size=6, opacity=0.5),
                    name='Constraint Predictions',
                    mode='markers',
                    error_y=dict(type='data', visible=True, array=c_pred_std, color='rgba(1,1,1,0.2)'),
                    legendgroup = '2'
                ),
                row=1+nc, col=1
            )

    ''' axis3 - Failure Predictions
    ----------------------------'''
    if l_pred_prob is not None:
        sel_true  = ((l_pred_prob>=0.5) & (l>=0.5)) | ((l_pred_prob<=0.5) & (l<=0.5))
        sel_false = ~sel_true
        fig.add_trace(
            go.Scatter(
                x=np.argwhere(sel_true)[:,0], y=l_pred_prob[sel_true],
                marker=dict(color='blue', size=10, symbol='circle-open'), # , opacity=0.5
                name='Predictive Probability of Success - True Positive/True Negatives',
                mode='markers',
                legendgroup = '3'
            ),
            row=1+nc+1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=np.argwhere(sel_false)[:,0], y=l_pred_prob[sel_false],
                marker=dict(color='red', size=10, symbol='circle-open'), # , opacity=0.5
                name='Predictive Probability of Success - False Positive/False Negatives',
                mode='markers',
                legendgroup = '3'
            ),
            row=1+nc+1, col=1
        )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(l))), y=l[:,0],
            marker=dict(color='black', size=6), # , opacity=0.5
            name='Evaluation',
            mode='markers',
            legendgroup = '3'
        ),
        row=1+nc+1, col=1
    )

    fig.update_layout(
        height = (nc+1)*800, width=1400, 
        title_text="Optimization Progress for Constrained Combinatorial Bayesian Optimization", 
        xaxis1_title='Number of Evaluations',
        yaxis1_title=f'objective function',
        **{f'yaxis{i+2}_title': f'constraint function #{i:d}'  for i in range(nc)},
        **{f'xaxis{i+2}_title': 'Number of Evaluations' for i in range(nc)},
        **{f'xaxis{1+nc+1}_title': 'Number of Evaluations'},
        **{f'yaxis{1+nc+1}_title': f'stability function'},
        xaxis_range=[-2,len(y)+2],
        legend_tracegroupgap = 450,
        xaxis_showticklabels=True, xaxis2_showticklabels=True
    )

    aim_run.track(aim.Figure(fig), step=0, name='optimization_progress')

    # fig.show()
    # return fig

def aim_plot_opt_graph(log:dict, aim_run:'aim.Run'=None, *args, **kwargs):
    ''' 
    References:
        - https://plotly.com/python/network-graphs/
        - https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html
        - https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_adjacency.html
    '''
    # find Adjacency Matrix of the Hamming Graph
    X1, X2 = [log['X'].astype(int)] * 2
    diff = X1[:,None] - X2[None,:]
    diff[np.abs(diff) > 1e-5] = 1
    hamming_dist = diff.sum(axis=-1)
    adj_matrix = hamming_dist.copy()
    adj_matrix[adj_matrix > 1] = 0

    # X = [tuple(x) for x in log.X]
    # d = log.X.shape[1]
    # dod = {i: {j: {"weight": min(d-hamming_dist[i,j], 5) } for j,xj in enumerate(X)} for i,xi in enumerate(X)}
    # G_hamm = nx.from_dict_of_dicts(dod)
    
    G = nx.from_pandas_adjacency(df=pd.DataFrame(data=adj_matrix))

    layout = nx.spring_layout(G, weight=0.01, k=0.2)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    node_x = []
    node_y = []
    for node in G.nodes():
        _x, _y = layout[node]
        node_x.append(_x)
        node_y.append(_y)
    text = []
    for _x,_y in zip(log['X'], log['y']):
        text.append( f'{_x}: {_y}' )  # list(zip(X.tolist(), y.tolist()))

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y,
            text = text,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='Hot',
                reversescale=True,
                color=log['y'][:,0],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Objective function',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
    )
    fig.update_layout(
        height = 800, width=800, 
        title_text="Combinatorial function evaluation graph", 
    )
    # fig.show()
    n_evals = len(log['X'])
    aim_run.track(aim.Figure(fig), step=n_evals, name='optimization_graph')
    # return fig

def aim_track_opt_metrics(log:dict, aim_run:'aim.Run', *args, **kwargs):
    y_min = np.fmin.accumulate(log['y'])
    y_min_feas = np.fmin.accumulate( np.where((log['c'] <= 0).all(1), log['y'], np.nan) )
    # calculate the cumulative mean of log['l'] 
    l_mean = np.cumsum(log['l'], axis=0) / np.arange(1,len(log['l'])+1)[:,None]
    
    y_pred_mean = log.get('y_pred_mean', np.nan*np.ones_like(log['y']))
    c_pred_mean = log.get('c_pred_mean', np.nan*np.ones_like(log['c']))
    
    y_pred_NMAE = np.abs(log['y'] - y_pred_mean) / (1e-4 + np.nanstd(log['y']))
    c_pred_NMAE = np.abs(log['c'] - c_pred_mean) / (1e-4 + np.nanstd(log['c'], axis=0))

    # track metrics per number of evaluations
    for i in range(len(y_min)):
        aim_run.track(
            dict(
                y_min 	    = y_min[i,0],
                y_min_feas  = y_min_feas[i,0],
                y_pred_NMAE = 0,
                **{f'c_pred_NMAE_{i:d}':ci for i,ci in enumerate(c_pred_NMAE[i])},
                l_mean	    = l_mean[i,0]
            ),
            step=i
        )

def aim_plot_equation_discovery_improvement(
    log:dict, iter_new_evals:List[int], aim_run:'aim.Run', 
    model:Sparse_Identification_Nonlinear_Dynamical_Systems,
    nbr_init_samples:int=np.inf,
    *args, **kwargs
):
    (idx_feas,) = np.where((log['c'] <= 0).all(1))
    if len(idx_feas) == 0:
        return
    idx_min = idx_feas[np.nanargmin(log['y'][idx_feas])]
    N = len(log['y'])
    # record image if we have just found a new minimum
    if idx_min >= iter_new_evals.min():
        fig = model.plot(X = log['X'][None,idx_min])[0]
        fig.suptitle(f"x: {log['X'][idx_min]}  y: {log['y'][idx_min,0]:.2f} c: {log['c'][idx_min,0]:.2f}\niteration: {idx_min}")
        fig.tight_layout()
        plt.close(fig)
        aim_run.track( aim.Image(fig), name='best_model_simulation', step=int(idx_min) )


''' DEPRECATED
--------------------'''

def plotly_plot_sigma2_Torch(sigma2_prior, sigma2_post):
    sigma2_prior_scipy = stats.invgamma(
        a     = sigma2_prior.concentration.detach().numpy(), 
        loc   = 0, 
        scale = sigma2_prior.rate.detach().numpy()
    )
    sigma2_post_scipy = stats.invgamma(
        a     = sigma2_post.concentration.detach().numpy(), 
        loc   = 0, 
        scale = sigma2_post.rate.detach().numpy()
    )
    return plotly_plot_sigma2(sigma2_prior_scipy, sigma2_post_scipy)

def plotly_plot_sigma2(sigma2_prior, sigma2_post):

    s = np.linspace(0, 1, 100)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Prior', 'Posterior'))
    fig.add_trace(
        go.Scatter(
            x=s, y=sigma2_prior.pdf(s),
            line=dict(color='blue'),
            name=r'$pdf(\sigma^2)$',
            legendgroup = '1',
            legendgrouptitle_text='Prior',
            mode='lines'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=s, y=sigma2_post.pdf(s),
            line=dict(color='orange'),
            name=r'$pdf(\sigma^2|y)$',
            legendgroup = '1',
            mode='lines'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=s, y=sigma2_prior.cdf(s),
            line=dict(color='purple'),
            name=r'$cdf(\sigma^2)$',
            legendgroup = '2',
            legendgrouptitle_text='Posterior',
            mode='lines'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=s, y=sigma2_post.cdf(s),
            line=dict(color='red'),
            name=r'$cdf(\sigma^2|y)$',
            legendgroup = '2',
            mode='lines'
        ),
        row=1, col=2
    )
    fig.update_layout(
        height=600, width=1200, 
        title_text="Model Error Estimation", 
        xaxis1_title=r'$\sigma^2$',
        xaxis2_title=r'$\sigma^2$',
        # yaxis1_title=r'$f(x)$',
        xaxis1_range=[0,1],
        xaxis2_range=[0,1],
        # legend_tracegroupgap = 20,
        # legend=dict(
        #     groupclick="toggleitem",
        #     yanchor="top",
        #     y=0.1,
        #     xanchor="right",
        #     x=1.5
        # )
    )
    # fig.show()
    return fig

def plot_opt_progress(X, y, c=None, y_all_best=None, y_all_best_feas=None, y_pred_mean=None, y_pred_std=None):

    plt.rcParams['lines.linewidth'] = 1.5

    fig, axs = plt.subplots(2,1, figsize=(8,10))
    axs[0].scatter(range(len(y)), y, c='tab:red', s=6**2, marker=MarkerStyle('o', 'none', ), label='Evaluations')
    axs[0].step(range(len(y)), np.minimum.accumulate(y), "-", where='post', label=f"Best", color='tab:blue', zorder=98)
    axs[0].errorbar(
        range(len(y)), y=y_pred_mean, yerr=y_pred_std,
        marker=MarkerStyle('o', 'none'), ms=6, ls="", c='tab:gray', alpha=0.3, capsize=3, capthick=3, label='Predictions',  markeredgewidth=2 #, lw=2, uplims=True, lolims=True
    )
    if y_all_best:
        axs[0].plot([-1e8, 1e8], [y_all_best,y_all_best], "--", color='k', label='Optimum')
    axs[0].set_ylabel(r'$f(x)$')
    axs[0].set_xlabel('Number of Evaluations')
    axs[0].set_title('Optimization Progress for All Configurations')
    # axs[0].set_xlim([min(n_evals),max(n_evals)])
    axs[0].set_ylim([min(y)-np.ptp(y)*0.1 if not y_all_best else y_all_best-np.ptp(y)*0.1, max(y)+np.ptp(y)*0.1])
    axs[0].set_xlim([-2, len(y)+2])
    axs[0].legend()
    axs[0].grid()

    if c is not None:
        
        idx_feas = np.where((c <= 0).all(1))[0]
        
        axs[1].scatter( idx_feas, y[idx_feas], c='tab:red', s=6**2, marker=MarkerStyle('o', 'none', ), label='Feasible evaluations')
        axs[1].step( idx_feas, np.minimum.accumulate(y[idx_feas]) , "-", where='post', label=f"Best", color='tab:blue', zorder=98)
        axs[1].errorbar(
            idx_feas, y=y_pred_mean[idx_feas], yerr=y_pred_std[idx_feas],
            marker=MarkerStyle('o', 'none'), ms=6, ls="", c='tab:gray', alpha=0.3, capsize=3, capthick=3, label='Predictions',  markeredgewidth=2 #, lw=2, uplims=True, lolims=True
        )
        if y_all_best_feas:
            axs[1].plot([-1e8, 1e8], [y_all_best_feas,y_all_best_feas], "--", color='k', label='Feasible optimum')
        axs[1].set_ylabel(r'$f(x)$')
        axs[1].set_xlabel('Number of Evaluations')
        axs[1].set_title('Optimization Progress for Feasible Configurations')
        # axs[0].set_xlim([min(n_evals),max(n_evals)])
        axs[1].set_ylim([min(y)-np.ptp(y)*0.1 if not y_all_best else y_all_best-np.ptp(y)*0.1, max(y)+np.ptp(y)*0.1])
        axs[1].set_xlim([-2, len(y)+2])
        axs[1].legend()
        axs[1].grid()

    plt.tight_layout()
    # plt.show()
    return fig

def plot_opt_progress_constraints(c=None, c_pred_mean=None, c_pred_std=None):

    plt.rcParams['lines.linewidth'] = 1.5

    if c_pred_mean is not None and c_pred_std is not None:
        ic = 0
        fig, axs = plt.subplots(1,1, figsize=(8,5))
        axs = [axs]
        axs[0].scatter(range(len(c)), c[:,ic], c='tab:red', s=6**2, marker=MarkerStyle('o', 'none', ), label=f'Constraint evaluations')
        axs[0].errorbar(
            range(len(c)), y=c_pred_mean[:,ic], yerr=c_pred_std[:,ic],
            marker=MarkerStyle('o', 'none'), ms=6, ls="", c='tab:gray', alpha=0.3, capsize=3, capthick=3, label='Predictions',  markeredgewidth=2 #, lw=2, uplims=True, lolims=True
        )
        axs[0].set_ylabel(r'$g(x)$')
        axs[0].set_xlabel('Number of Evaluations')
        axs[0].set_title(f'Constraints [{ic}]')
        # axs[0].set_xlim([min(n_evals),max(n_evals)])
        axs[0].set_ylim([min(c[:,ic])-np.ptp(c[:,ic])*0.1, max(c[:,ic])+np.ptp(c[:,ic])*0.1])
        axs[0].set_xlim([-2, len(c[:,ic])+2])
        axs[0].legend()
        axs[0].grid()

    return fig

