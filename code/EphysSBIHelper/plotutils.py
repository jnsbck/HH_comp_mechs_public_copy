import numpy as np
import pandas as pd
import torch

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from EphysSBIHelper.simutils import runHH
from EphysSBIHelper.evalutils import generate_correlated_parameters, compare_correlated_summary_stats

colormap = mpl.cm.winter # defines colormap for the following functions

def plot_summary_stats(stats_sim, stats_obs=None, savefig=False, fig_name="default"):
    """Plot histograms for each summary statistic of the simulated data and
    compares it to the summary statistics of the observed voltage trace.
    Adpated from code courtesy of @ybernaerts.

    Parameters
    ----------
    stats_sim : pd.DataFrame
        Summary statistics of simulated data.
    stats_obs : pd.DataFrame
        Summary statistics of observed data.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure."""

    fig, axes = plt.subplots(6, 5, figsize = (12, 12))
    axes = axes.reshape(-1)

    for ax, col in zip(axes, stats_sim.columns):
        if col == 'AP average amp adapt':
            selected_vals = stats_sim[col][(stats_sim[col] > 0.45) & (stats_sim[col] < 0.55)]
            ax.hist(selected_vals, zorder=0, alpha=0.5, color='grey', bins=30)
        elif col == 'ISI average adapt':
            selected_vals = stats_sim[col][(stats_sim[col] > 0.48) & (stats_sim[col] < 0.52)]
            ax.hist(selected_vals, zorder=0, alpha=0.5, color='grey', bins=30)
        elif col == 'SFA':
            selected_vals = stats_sim[col][stats_sim[col] < 1.1]
            ax.hist(selected_vals, zorder=0, alpha=0.5, color='grey', bins=30)
        elif col == 'ISI adapt':
            selected_vals = stats_sim[col][(stats_sim[col] > -0.8) & (stats_sim[col] < 0.8)]
            ax.hist(selected_vals, zorder=0, alpha=0.5, color='grey', bins=30)
        elif col == r'$V_{m}$ turtosis':
            selected_vals = stats_sim[col][stats_sim[col] < 30]
            ax.hist(selected_vals, zorder=0, alpha=0.5, color='grey', bins=30)
        elif col == r'rest $V_{m}$ std':
            selected_vals = stats_sim[col][stats_sim[col] < 0.1]
            ax.hist(selected_vals, zorder=0, alpha=0.5, color='grey', bins=30)
        else:
            ax.hist(stats_sim[col], zorder=0, alpha=0.5, color='grey', bins=30)

        if type(stats_obs) != type(None):
            ax.vlines(stats_obs[col][0], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], zorder=1, color='blue')
        ax.set_xlabel(col)

    # removes axis from unused subplots -> essentially hides them
    for i, ax in enumerate(axes[len(stats_sim.columns):]):
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        sns.despine(ax = ax, left = True, bottom = True)

    plt.tight_layout()
    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_summ_stats_Iscale1.png'.format(fig_name))

def plot_parameters(parameter_samples, est_obs_params, figsize=(9, 9), savefig=False, fig_name="default"):
    """Plot histogram of parameter samples from the posterior distribution
    and mark position of estimated observed parameters.

    Parameters
    ----------
    parameter_samples : ndarray
        The parameter samples from the posterior distribution.
    est_obs_params : dict
        The parameter names and values for the parameters estimated
        to match the observed voltage trace well.
    figsize : tuple, list or ndarray
        Changes the figure size of the plot.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure."""

    fig, axes = plt.subplots(3, 3, figsize = figsize)

    for i, (ax, name, value) in enumerate(zip(axes.reshape(-1), est_obs_params.keys(), est_obs_params.values())):
        ax.hist(np.array(parameter_samples[:,i]), zorder=0, alpha=0.5, color='grey', bins=30)
        if type(est_obs_params) != None:
            ax.vlines(value, ymin=ax.get_ylim()[0], zorder=1, ymax=ax.get_ylim()[1], color='blue')
        ax.set_xlabel(name)
    plt.tight_layout()
    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_param_samples.png'.format(fig_name))


def plot_comparison(simulated_traces, trace_obs, samplesize=10, figsize=(10, 14), timewindow=None, savefig=False, fig_name="default", effect_on="traces",selected_stats=None):
    """Plot a sample of the simulated traces and compare it to the observed trace.

    Parameters
    ----------
    simulated_traces : Trace or list[Trace]
        The simulated data of the voltage traces in the form of Trace() objects.
    trace_obs : Trace
        Trace object for the observed voltage trace.
    samplesize : int
        How many simulated traces to compare.
    figsize : tuple or ndarray
        Changes the figure size of the plot.
    timewindow: tuple, list or ndarray
        Contains start and end time of voltage traces that will be shown.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure.
    effect_on : str ("traces" or "summary stats")
        Specify how the effect of a correlated change in parameters affects either the summary
        statistics or the voltage traces.
    selected_stats : None, tuple, list or array
        Indices of summary stats to be used."""

    if effect_on == "traces":
        if type(simulated_traces) == list or type(simulated_traces) == np.ndarray:
            fig, axes = plt.subplots(int((samplesize+1)/2), 2, figsize = figsize)

            for i, (ax, sim_result) in enumerate(zip(axes.reshape(-1), simulated_traces)):
                sim_result.inspect(axes=ax, label="simulated", voltage_only=True)
                trace_obs.inspect(axes=ax, label="observed", voltage_only=True, timewindow=timewindow)
                ax.set_ylabel('Membrane voltage [mV])', fontsize = 15)
                ax.set_xlabel('Time [ms]', fontsize = 15)
                ax.set_title('Simulation {}'.format(i), fontsize = 17)
            plt.tight_layout()
        else:
            fig = plt.figure(figsize = figsize)
            ax = plt.subplot(111)
            trace_obs.inspect(axes=ax, label="simulated", voltage_only=True)
            simulated_traces.inspect(axes=ax, label="observed", voltage_only=True, timewindow=timewindow)
            ax.set_ylabel('Membrane voltage [mV])', fontsize = 15)
            ax.set_xlabel('Time [ms]', fontsize = 15)
        plt.tight_layout() # not tested
    if effect_on == "summary_stats":
        n_stats = 0
        if selected_stats != None:
            n_stats = len(selected_stats)
        else:
            n_stats = len(trace_obs.summarise().values())
        fix, axes = plt.subplots(int(n_stats/4),int(n_stats/5), figsize=(12,10))
        axes = axes.reshape(-1)
        sim_summaries = [trace.summarise(selected_stats) for trace in simulated_traces]
        sim_summary_df = pd.DataFrame(summaries)
        summary_change_df = (summary_df-summary_df.iloc[0])/summary_df*100
        for i, col in enumerate(summary_change_df.columns):
            change = summary_change_df[col]
            change.plot(ax=axes[i], kind="bar")
            axes[i].hlines(0, xmin=0, xmax=len(change), colors="black")
            axes[i].set_title(col)
            axes[i].set_xlabel("n")
            axes[i].set_ylabel(r"$\Delta$ [%]")
            lim = np.max(abs(change.values))
            axes[i].set_ylim([-1.1*lim-1e-2,1.1*lim+1e-2])
        plt.tight_layout()

    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_sim_v_obs.png'.format(fig_name))


def plot_best_matches(thetas=None, stats_sim=None, stats_obs=None, simulated_traces=None,
                     trace_obs=None, samplesize=10, figsize=(10, 14), savefig=False,
                     fig_name="default", selected_stats=None, metric="MSE", mode="auto"):
    """Plot the best matching voltage traces compared with the observed trace,
    based on the mean squared error of the summary statistics.

    Parameters
    ----------
    thetas : torch.tensor or numpy.ndarray
        Holds the parameters that go along with stats_sim.
    stats_sim = pandas.DataFrame
        Contains the summarised simulation results.
    stats_obs = pandas.DataFrame
        Contains the summarised observation results.
    simulated_traces : list[Trace]
        The simulated data of the voltage traces in the form of Trace() objects.
    trace_obs : Trace
        Trace object for the observed voltage trace
    selected_stats : None, tuple, list or array
            Indices of summary stats to be used
    samplesize : int
        How many simulated traces to compare.
    figsize : tuple, list or ndarray
        Changes the figure size of the plot.
    timewindow: tuple, list or ndarray
        Contains start and end time of voltage traces that will be shown.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure.
    selected_stats : None, tuple, list or array
        Indices of summary stats to be used.
    metric : str
        Determines which metric to sort the samples by. Currently "MSE"
        and "std" are supported.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace")."""

    mins_idx, mins = best_matches(stats_sim, stats_obs, simulated_traces, trace_obs, selected_stats, metric)
    if type(simulated_traces) == type(None):
        if type(thetas) == pd.core.frame.DataFrame:
            thetas = torch.tensor(thetas.values[mins_idx[:samplesize]])
        simulated_traces = runHH(thetas, trace_obs, mode=mode)
    if trace_obs == None:
        raise ValueError("No observed trace was provided. Please provide Trace().")
    plot_comparison(simulated_traces, trace_obs, samplesize, figsize=(10, 14))



    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_best_sims_v_obs.png'.format(fig_name))

def plot_correlation_effects(params, syn_current, corr_mat, mag_of_change=0.1, figsize=(15,10),
                             timewindow=[0.3,0.33], effect_on="traces", selected_stats=None,
                             savefig=False, fig_name="default", summary_func="v1",
                             mode="auto"):
    """Plots the effects of changing correlated parameter pairs, for all pairs, on the simulated
    voltage traces or the change in summary statistics. Starts with (0,1).

    Parameters
    ----------
    params : torch.Tensor
        A 1D parameter tensor that holds the values of input parameters to the simulator model.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    corr_mat : torch.Tensor
        Matrix containing the pairwise correlation coefficients of the different parameters.
    mag_of_change : float
        Value from the intervall of [-k, k], where k \in R that specifies by how much the
        input parameters are altered.
    figsize : tuple
        Specify size of plt.figure.
    timewindow : tuple, list or ndarray
        The voltage and current trace will only be plotted between (t1,t2).
        To be specified in secs.
    effect_on : str ("traces" or "summary stats")
        Specify how the effect of a correlated change in parameters affects either the summary
        statistics or the voltage traces.
    selected_stats : None, tuple, list or array
        Indices of summary stats to be used.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure.
    summary_func : str
        A descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace")."""

    N = np.max(params.shape)

    fig, axes = plt.subplots(N-1,N-1, figsize=figsize)
    # loop over pairs in upper triangle of matrix
    for j in range(1,N):
        for i in range(j):
            pars = generate_correlated_parameters(params, corr_mat, pair=(i,j))
            correlated_trace = runHH(pars, syn_current=syn_current, mode=mode)
            base_trace = runHH(params, syn_current=syn_current, mode=mode)

            axs = axes[i,j-1]

            if "trace" in effect_on.lower():
                # plot traces
                base_trace.inspect(axes=axs, voltage_only=True, timewindow=timewindow)
                correlated_trace.inspect(axes=axs, voltage_only=True, timewindow=timewindow)

                # format axes
                if i != j-1:
                    axs.get_xaxis().set_ticks([])
                    axs.get_yaxis().set_ticks([])
                    axs.set_xticklabels([])
                    axs.set_yticklabels([])
                    axs.set_xlabel(None)
                    axs.set_ylabel(None)

                    plt.subplots_adjust(hspace = .65, wspace = .2)

                else:
                    if i == 0:
                        axs.set_ylabel("V [mV]", fontsize=10)
                        axs.set_xlabel("j = "+str(j), fontsize=10)
                    if j == 7:
                        axs.set_xlabel("t [ms]", fontsize=10)
                        axs.set_ylabel("i = "+ str(i), fontsize=10)
                    if i != 0 and j != 7:
                        axs.set_xlabel("j = "+ str(j), fontsize=10)
                        axs.set_ylabel("i =" + str(i), fontsize=10)
            if "summary" in effect_on.lower():
                summary_df = compare_correlated_summary_stats(params, syn_current, (i,j), corr_mat, mag_of_change, selected_stats, summary_func=summary_func, mode=mode)
                changes_df = pd.DataFrame(abs((summary_df["base params"] - summary_df["correlated params"])/summary_df["base params"])*100, columns=[r"$\Delta_{corr}$"+" = {}\%".format(mag_of_change*100)])

                # plot changes
                changes_df.plot(kind="bar", ax=axs, color="black",legend=False, width=1)

                # format axes and plots
                axs.set_xticklabels([])
                axs.set_yticklabels([])
                axs.get_xaxis().set_ticks([])
                axs.get_yaxis().set_ticks([])
                axs.set_ylim([0,100])

                plt.subplots_adjust(hspace = .65, wspace = .2)
                plt.suptitle("Percentage change in the summary statistics")

                if i == j-1:
                    axs.set_xlabel("j = "+ str(j), fontsize=10)
                    axs.set_ylabel("i =" + str(i), fontsize=10)
                    axs.get_yaxis().set_ticks([0,100])
                    axs.set_yticklabels([0,100])
                    if i == 0 and j == 1:
                        axs.set_ylabel(r"$\Delta$ [%]", fontsize=10)
                        axs.set_xlabel("j = "+str(j), fontsize=10)
                    if j == N-1 and i == N-2:
                        axs.set_xlabel("Summary Stats", fontsize=10)
                        axs.set_ylabel("i = "+ str(i), fontsize=10)

                else:
                    axs.set_xlabel("j = "+ str(j), fontsize=10)
                    axs.set_ylabel("i =" + str(i), fontsize=10)

    # remove diagonal and lowever triangle of matrix
    for i in range(1,N):
        for j in range(i-1):
            axs = axes[i-1,j]
            axs.get_xaxis().set_ticks([])
            axs.get_yaxis().set_ticks([])
            sns.despine(ax = axs, left = True, bottom = True)
    plt.suptitle("")

    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_corr_change_effect_on_{}.png'.format(fig_name, effect_on))

def plot_change_of_corr_summaries(sum_df, changes_df, color="black", savefig=False, fig_name="default"):
    """Plot the change and relative change in summary statistics between
    a reference trace and another trace.

    Parameters
    ----------
    sum_df : pandas.DataFrame
        Contains the summary statistics for both the base and correlated trace.
    changes_df : pandas.DataFrame
        Contains the relative difference between two sets of summary statistics.
    color : str
        Specify color of barplot.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure."""

    fig, ax = plt.subplots(1,2, figsize=(12,5))

    sum_df.plot(kind="bar", ax=ax[0], color=["grey", color])

    changes_df.plot(kind="bar", ax=ax[1], color=color, legend=False)
    ax[1].set_ylabel(r"$\Delta_{param}$ [%]")
    ax[1].set_ylim([0,50])

    plt.suptitle(changes_df.columns[0])
    plt.tight_layout()

    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_corr_change_effect_on_summaries.png'.format(fig_name))

def plot_correlated_summary_stats(params, syn_current, corr_mat, pair=(0,1), start=0.01,
                                  stop=0.5, N=5, figsize=(15,10), timewindow=[0.3,0.33],
                                  selected_stats=None, change_only=False,
                                  savefig=False, fig_name="default",
                                  mode="auto", summary_func="v1"):
    """Plots the effects of changing correlated parameters, for one parameter pair,
    on the summary stats of the simulated voltage traces for different magnitudes of change.

    Parameters
    ----------
    params : torch.Tensor
        A 1D parameter tensor that holds the values of input parameters to the simulator model.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    corr_mat : torch.Tensor
        Matrix containing the pairwise correlation coefficients of the different parameters.
    pair : tuple, list, numpy.ndarray
        Specifies which correlation axis to change the parameters along.
    start : float
        Minimum value of parameter change along correlation axis.
    stop : float
        Maximum value of parameter change along correlation axis.
    N : int
        Number of values inbetween min. and max. amount of change.
    figsize : tuple
        Specify size of each plt.figure.
    timewindow : tuple, list or ndarray
        The voltage and current trace will only be plotted between (t1,t2).
        To be specified in secs.
    selected_stats : None, tuple, list or array
        Indices of summary stats to be used.
    change_only : bool
        Whether or not to only plot the change in the difference of summary stats.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace")."""

    base_trace = runHH(params, syn_current=syn_current, mode=mode)
    base_trace.summarise(selected_stats)

    change_hist = pd.DataFrame(data=None,columns=base_trace.Summary.keys())
    for idx, change in enumerate(np.linspace(start, stop, N)):
        sum_df = compare_correlated_summary_stats(params, syn_current, pair, corr_mat, change, selected_stats, mode=mode, summary_func=summary_func)

        rel_changes = abs((sum_df["base params"] - sum_df["correlated params"])/sum_df["base params"])*100
        changes_df = pd.DataFrame(rel_changes, columns=[r"$\Delta_{corr}$"+" = {}%".format(change*100)])

        change_hist = change_hist.append(changes_df.T)

        if not change_only:
            plot_change_of_corr_summaries(sum_df, changes_df, colormap(idx/N), savefig, fig_name)

    change_hist.T.plot(kind="bar", figsize=(15,5), cmap=colormap)
    plt.ylim(0,50)

    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_corr_change_effect_on_summaries_change.png'.format(fig_name))


def show_correlated_traces(params, syn_current, corr_mat, pair=(0,1), start=0.01, stop=0.5,
                           N=5, figsize=(15,10), timewindow=[0.3,0.33], compare_changes=False,
                           savefig=False, fig_name="default",
                           mode="auto"):
    """Plots the effects of changing correlated parameters, for one parameter pair,
    on the summary stats of the simulated voltage traces for different magnitudes of change.

    Parameters
    ----------
    params : torch.Tensor
        A 1D parameter tensor that holds the values of input parameters to the simulator model.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    corr_mat : torch.Tensor
        Matrix containing the pairwise correlation coefficients of the different parameters.
    pair : tuple, list, numpy.ndarray
        Specifies which correlation axis to change the parameters along.
    start : float
        Minimum value of parameter change along correlation axis.
    stop : float
        Maximum value of parameter change along correlation axis.
    N : int
        Number of values inbetween min. and max. amount of change.
    figsize : tuple
        Specify size of each plt.figure.
    timewindow : tuple, list or ndarray
        The voltage and current trace will only be plotted between (t1,t2).
        To be specified in secs.
    compare_changes : bool
        Whether or not to include a plot that compares the correlated traces for all
        magnitudes of change.
    savefig : bool
        Determines whether or not to save the plotted figure.
    fig_name : str
        Integrates figure name into the filename of the saved figure.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace")."""

    base_trace = runHH(params, syn_current=syn_current, mode=mode)
    corr_traces = []

    fig = plt.figure(figsize=figsize)
    for idx, change in enumerate(np.linspace(start, stop, N)):

        # generate correlated params, simulate and store trace
        pars = generate_correlated_parameters(params, corr_mat, pair=pair, mag_of_change=change)
        corr_trace = runHH(pars, syn_current=syn_current)
        corr_traces.append((change,corr_trace))

        # plot both traces onto same axis
        axs = base_trace.inspect(voltage_only=True, timewindow=timewindow, label="base params",
                                 line_color="grey")
        corr_trace.inspect(axes=axs, voltage_only=True, timewindow=timewindow, label="correlated params",
                           line_color=colormap(idx/N))

        plt.ylim(min(base_trace.Vt*1e3),max(base_trace.Vt*1e3))
        plt.legend(loc=1)
        plt.title("change along correlation axis = {0:.1f}%".format(change*100))

    if savefig:
        plt.savefig('../data/figures/{}_8paramprior_support_corr_change_effect_on_traces_{}.png'.format(fig_name, pair))

    # plots all changes into one set ox axes
    if compare_changes:
        axs = base_trace.inspect(voltage_only=True, timewindow=timewindow, label="base params",
                                 line_color="grey")
        for idx, (change,corr_trace) in enumerate(corr_traces):
            corr_trace.inspect(axes=axs, voltage_only=True, timewindow=timewindow,
                               label="{0:.1f}%".format(change*100), line_color=colormap(idx/N))
        plt.legend(loc=1)

        if savefig:
            plt.savefig('../data/figures/{}_8paramprior_support_corr_change_effect_on_traces_combined_{}.png'.format(fig_name, pair))
