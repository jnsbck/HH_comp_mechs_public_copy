import numpy as np
import pandas as pd
import torch
#import importlib # probably not needed

# import of data
from scipy.io import loadmat

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# some ephys extraction
import EphysSBIHelper.ephys_extractor as efex
import EphysSBIHelper.ephys_features as ft

# to calculate moments
from scipy import stats as spstats

# to fit R_in
from scipy.optimize import curve_fit

# Functions
def sigmoid(x, offset = 1, steepness = 1):
    """Applies the sigmoid function

    y = 1/(1+e^{-s*(x-x_0)})

    to an input x.

    Parameters
    ----------
    x : float
    offset : float
    steepness : float

    Returns
    -------
    y : float"""

    # offset to shift the sigmoid centre to 1
    return 1/(1 + np.exp(-steepness*(x-offset)))

def normalise_df(df):
    """Normalise DataFrame along its columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be normalise along its columns.

    Returns
    -------
    normalised_df: pd.DataFrame
        Input DataFrame normlised along the columns."""
    m = df.mean()
    s = df.std()
    normalised_df = (df - m)/s
    return normalised_df

# Objects

class Trace:
    """Aggregates the electrophysiological data from single current clamp experiment
    and provides methods for easy analysis, visualisation and simulation/inference
    procedures.

    Parameters
    ----------
    trace : dict
        Contains the membrane voltage trace along with the time data under the "data" key, possibly the a
        description string of the trace and relevant experimental parameters like
        stimulus onset, duration and end."""

    def __init__(self, trace=None):
        self.Vt = None # [V]
        self.ts = None # [s]
        self.Bins = None
        self.It = None # [A/m^2]
        self.ItA = None #[m^2]

        self.dt = None # [s]
        self.SampFq = None # [Hz]

        self.Idx = None
        self.IIdx = None
        self.IIn = None # [A]
        self.ElecIdx = None
        self.Description = None

        self.StimOnset = None
        self.StimDur = None
        self.StimEnd = None

        self.Summary = {}

        if trace != None:
            if "data" in trace.keys():
                self.set_trace(trace["data"])
            if "description" in trace.keys():
                self.set_trace_attrs(trace["description"])
            if "params" in trace.keys():
                self.set_trace_parameters(trace["params"])

    ## getters and setters
    def get_summary_stats(self):
        """Retrieve the summary statistics."""

        return self.Summary

    def set_summary_stats(self, dct):
        """Sets the summary statistics.

        Parameters
        ----------
        dct : dict
            Contains names and values of summary statistics."""

        self.Summary = dct

    def set_trace(self, trace):
        """Set voltage and possibly time data.

        Parameters
        ----------
        trace : ndarray
            Either 1D or 2D, holding either just the mebrane voltage trace or the
            time points as well."""

        if trace.ndim > 1:
            self.ts, self.Vt = trace.T
            self.dt = self.ts[1]-self.ts[0]
            self.NumBins = len(self.ts)
            self.Bins = np.arange(self.NumBins)
            self.SampFq = 1/self.dt
        else:
            self.Vt = trace

    def set_trace_parameters(self, params):
        """Set member variables / trace parameters.

        Parameters
        ----------
        params : dict
            Key-value pairs for member variables and there values."""

        for key in params.keys():
            self.__dict__[key] = params[key]
            # if key in self.__dict__.keys(): # LOOKS LIKE IT IS REDUNDANT
            #     self.__dict__[key] = params[key]

    def set_trace_attrs(self, description):
        """Set trace parameters according to descriptive string from data.

        Parameters
        ----------
        description : str
            String of the form "Trace_X_X_X_X", where X stands for a digit 1-9."""

        self.Description = description
        attrs = list(map(int, description.split("_")[1:])) # splitting string and converting digits to int
        self.ElecIdx = attrs[-1]
        self.Idx = attrs[1]
        self.IIdx = attrs[2]

    def get_syn_current(self):
        """Retrieve the current clamp protocol for this Trace.

        Returns
        -------
        syn_current : dict
            Contains the initial current "V0" [float], the stimulation current trace
            "It" [ndarray], the product of the stimulation current trace and the area
            of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
            "dt": float and the simulus time course "StimOnset": float,
            "StimEnd": float and "StimDur": float.

            {"V0": [float], "It": [ndarray], "ItA": [ndarray], "ts": [ndarray], "dt":[float],
                    "StimOnset": [float], "StimEnd": [float], "StimDur": [float]}"""

        syn_current = {"V0":np.mean(self.Vt[:2500]), "It":self.It, "ItA": self.ItA, "ts":self.ts, "dt":self.dt,
                "StimOnset": self.StimOnset, "StimEnd": self.StimEnd, "StimDur": self.StimDur}

        return syn_current

    # methods
    def __str__(self):
        """Returns self.__dict__ for printing Trace objects."""

        return str(self.__dict__)

    def inspect(self, axes=None, figsize=(12, 4), voltage_only=False, title="", label="", timewindow=None, savefig=False, fig_name="default", line_color=None):
        """Plot voltage and possibly current trace.

        Parameters
        ----------
        axes : plt.Axes
            The set of axes this plot will be added to.
        figsize : tuple, list or ndarray
            Changes the figure size of the plot.
        voltage_only : bool
            Only the membrane voltage will be plotted.
        title : str
            Adds a custom title to the figure.
        label : str
            Adds a custom label to the trace.
        timewindow : tuple, list or ndarray
            The voltage and current trace will only be plotted between (t1,t2).
            To be specified in secs.
        savefig : bool
            Determines whether or not to save the plotted figure.
        fig_name : str
            Integrates figure name into the filename of the saved figure.

        Returns
        -------
        axes : plt.Axes
            The set of axes of this plot in order to add further
            plots to the same set of axes."""

        start, end = (0,-1)
        if timewindow != None:
            start, end = (np.array(timewindow)/self.dt).astype(int)

        Vt = self.Vt[start:end]*1e3
        ts = self.ts[start:end]*1e3
        It = np.zeros_like(ts)
        if type(self.It) != type(None):
                It = self.ItA[start:end]*1e9

        ax1,ax2 = (0,0)
        if voltage_only:
            if axes == None:
                fig = plt.figure(figsize=figsize)
                axes = plt.subplot(111)

            axes.plot(ts, Vt, label=label, c=line_color);
            axes.set_xlabel('Time [ms]', fontsize = 14)
            axes.set_ylabel('Membrane voltage [mV]', fontsize = 14)
            if label != "":
                    axes.legend()

        else:
            if axes == None:
                fig = plt.figure(1,figsize=figsize)
                gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1], figure=fig)
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                axes = [ax1, ax2]
            if axes != None:
                ax1, ax2 = axes

            ax1.plot(ts, Vt, label=label, c=line_color)
            ax1.set_ylabel("V [mV]")
            ax1.set_title(title)
            if label != "":
                ax1.legend()
            #plt.setp(ax1, xticks=[], yticks=[-80, -20, 40])

            ax2.plot(ts, It, lw=2, c="black")
            ax2.set_xlabel("t [ms]")
            ax2.set_ylabel("I [nA]")

            #ax2.set_xticks([0, max(self.ts)/2, max(self.ts)])
            ax2.set_yticks([0, np.max(It)])
            ax2.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        if savefig:
            plt.savefig('../data/figures/{}_8paramprior_support_vtrace.png'.format(fig_name))

        return axes

    def calculate_summary_v2(self, selected_stats=None):
        """Calculate statistics that summarise the important features of trace object.

        Adpated from code courtesy of @ybernaerts.

        Parameters
        ----------
        selected_stats : None, tuple, list or array
            Indices of summary stats to be used.

        Returns
        -------
        summary statistics : dict
            Contains key-value pairs of the summary statistics.
        """

        n_mom = 3

        t = self.ts
        dt = self.dt
        V = self.Vt*1e3
        I = self.It

        # -------- #
        # 1st part: features that electrophysiologists are actually interested in #
        EphysObject = efex.EphysSweepFeatureExtractor(t = t, v = V, \
                                                      i = I, start = self.StimOnset, \
                                                      end = self.StimEnd, filter = 10)
        # -------- #
        # 1st part: features that electrophysiologists are actually interested in #
        EphysObject.process_spikes()
        AP_count = np.nan
        #fano_factor = np.nan
        cv = np.nan
        AI = np.nan
        #AI_adapt_average = np.nan
        latency = np.nan
        AP_amp_adapt = np.nan
        AP_amp_adapt_average = np.nan
        AHP = np.nan
        AP_threshold = np.nan
        AP_amplitude = np.nan
        AP_width = np.nan
        #UDR = np.nan
        AHP_3 = np.nan
        AP_threshold_3 = np.nan
        AP_amplitude_3 = np.nan
        AP_width_3 = np.nan
        #UDR_3 = np.nan
        #AP_fano_factor = np.nan
        AP_cv = np.nan
        SFA = np.nan

        if EphysObject._spikes_df.size:
            EphysObject._spikes_df['peak_height'] = EphysObject._spikes_df['peak_v'].values - \
                                                   EphysObject._spikes_df['threshold_v'].values
            AP_count = EphysObject._spikes_df['threshold_i'].values.size
        if not EphysObject._spikes_df.empty: # There are APs and in the positive current regime
            if False in list(EphysObject._spikes_df['clipped']): # There should be spikes that are also not clipped

                # Add the Fano Factor of the interspike intervals (ISIs), a measure of the dispersion of a
                # probability distribution (std^2/mean of the isis)
                #fano_factor = EphysObject._sweep_features['fano_factor']

                # Add the coefficient of variation (std/mean, 1 for Poisson firing Neuron)
                cv = EphysObject._sweep_features['cv']

                # And now the same for AP heights in the trace
                #AP_fano_factor = EphysObject._sweep_features['AP_fano_factor']
                AP_cv = EphysObject._sweep_features['AP_cv']

                # Adding spike frequency adaptation (ratio of spike frequency of second half to first half for the highest
                # frequency count trace)

                half_stim_index = ft.find_time_index(t, np.float(self.StimOnset + self.StimDur/2))
                if AP_count > 2: # We only consider traces with more than 8.333 Hz = 5/600 ms spikes here

                    AHP_3 = EphysObject._spikes_df.loc[2, 'fast_trough_v'] - EphysObject._spikes_df.loc[2, 'threshold_v']
                    AP_threshold_3 = EphysObject._spikes_df.loc[2, 'threshold_v']
                    AP_amplitude_3 = EphysObject._spikes_df.loc[2, 'peak_height']
                    AP_width_3 = EphysObject._spikes_df.loc[2, 'width']*1000
                    #UDR_3 = EphysObject._spikes_df.loc[2, 'upstroke_downstroke_ratio']
                    if np.sum(EphysObject._spikes_df['threshold_index'] < half_stim_index)!=0:
                        SFA = np.sum(EphysObject._spikes_df['threshold_index'] > half_stim_index) / \
                              np.sum(EphysObject._spikes_df['threshold_index'] < half_stim_index)

                # Add the (average) adaptation index
                AI = EphysObject._sweep_features['isi_adapt']
                #AI_adapt_average = EphysObject._sweep_features['isi_adapt_average']

                # Add the latency
                latency = EphysObject._sweep_features['latency']*1000

                # Add the AP amp (average) adaptation (captures changes in AP amplitude during stimulation time)
                AP_amp_adapt = EphysObject._sweep_features['AP_amp_adapt']
                AP_amp_adapt_average = EphysObject._sweep_features['AP_amp_adapt_average']


                # Add the AP AHP, threshold, amplitude, width and UDR (upstroke-to-downstroke ratio) of the
                # first fired AP in the trace
                AHP = EphysObject._spikes_df.loc[0, 'fast_trough_v'] - EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_threshold = EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_amplitude = EphysObject._spikes_df.loc[0, 'peak_height']
                AP_width = EphysObject._spikes_df.loc[0, 'width']*1000
                #UDR = EphysObject._spikes_df.loc[0, 'upstroke_downstroke_ratio']

        # -------- #
        # 2nd part: features that derive standard stat moments, possibly good to perform inference
        std_pw = np.power(
            np.std(V[(t > self.StimOnset) & (t < self.StimEnd)]), np.linspace(3, n_mom, n_mom - 2)
        )
        std_pw = np.concatenate((np.ones(1), std_pw))
        moments = (
            spstats.moment(
                V[(t > self.StimOnset) & (t < self.StimEnd)], np.linspace(2, n_mom, n_mom - 1)
            )
            / std_pw
        )

        rest_pot = np.mean(V[t < self.StimOnset])

        # concatenation of summary statistics
        features = np.array(['AP threshold', 'AP amplitude', 'AP width', 'AHP', \
                '3rd AP threshold', '3rd AP amplitude', '3rd AP width', '3rd AHP', \
                'AP count', 'AP amp adapt', \
                'AP average amp adapt', \
                'AP CV', 'ISI adapt', 'ISI CV', 'latency', 'SFA', \
                r'rest $V_{m}$ mean', r'$V_{m}$ mean', \
                r'$V_{m}$ std', r'$V_{m}$ skewness'])
        sum_stats_vec = np.concatenate(
            (
                np.array([AP_threshold, AP_amplitude, AP_width, AHP, \
                          AP_threshold_3, AP_amplitude_3, AP_width_3, AHP_3, \
                          AP_count, np.log(AP_amp_adapt), \
                          sigmoid(AP_amp_adapt_average, offset=1, steepness=50), \
                          np.log(AP_cv), np.log(AI), np.log(cv), np.log(latency+0.4), SFA]),
                np.array(
                    [rest_pot, np.mean(V[(t > self.StimOnset) & (t < self.StimEnd)])]
                ),
                moments,
            )
        )
        # sum_stats_vec = sum_stats_vec[0:n_summary]
        if selected_stats != None:
            summary_stats = dict(zip(features[selected_stats],sum_stats_vec[selected_stats]))
        else:
            summary_stats = dict(zip(features,sum_stats_vec))
        return summary_stats


    def calculate_summary_v1(self, selected_stats=None):
        """Calculate statistics that summarise the important features of trace object.

        Adpated from code courtesy of @ybernaerts.

        Parameters
        ----------
        selected_stats : None, tuple, list or array
            Indices of summary stats to be used.

        Returns
        -------
        summary statistics : dict
            Contains key-value pairs of the summary statistics.
        """

        n_mom = 3

        t = self.ts
        dt = self.dt
        V = self.Vt*1e3
        I = self.It


        # -------- #
        # 1st part: features that electrophysiologists are actually interested in #
        EphysObject = efex.EphysSweepFeatureExtractor(t = t, v = V, \
                                                      i = I, start = self.StimOnset, \
                                                      end = self.StimEnd, filter = 10)
        EphysObject.process_spikes()
        AP_count = np.nan
        fano_factor = np.nan
        cv = np.nan
        AI = np.nan
        AI_adapt_average = np.nan
        latency = np.nan
        AP_amp_adapt = np.nan
        AP_amp_adapt_average = np.nan
        AHP = np.nan
        AP_threshold = np.nan
        AP_amplitude = np.nan
        AP_width = np.nan
        UDR = np.nan
        AHP_5 = np.nan
        AP_threshold_5 = np.nan
        AP_amplitude_5 = np.nan
        AP_width_5 = np.nan
        UDR_5 = np.nan
        AP_fano_factor = np.nan
        AP_cv = np.nan
        SFA = np.nan

        if EphysObject._spikes_df.size:
            EphysObject._spikes_df['peak_height'] = EphysObject._spikes_df['peak_v'].values - \
                                                   EphysObject._spikes_df['threshold_v'].values
            AP_count = EphysObject._spikes_df['threshold_i'].values.size
        if not EphysObject._spikes_df.empty: # There are APs and in the positive current regime
            if False in list(EphysObject._spikes_df['clipped']): # There should be spikes that are also not clipped

                # Add the Fano Factor of the interspike intervals (ISIs), a measure of the dispersion of a
                # probability distribution (std^2/mean of the isis)
                fano_factor = EphysObject._sweep_features['fano_factor']

                # Add the coefficient of variation (std/mean, 1 for Poisson firing Neuron)
                cv = EphysObject._sweep_features['cv']

                # And now the same for AP heights in the trace
                AP_fano_factor = EphysObject._sweep_features['AP_fano_factor']
                AP_cv = EphysObject._sweep_features['AP_cv']

                # Adding spike frequency adaptation (ratio of spike frequency of second half to first half for the highest
                # frequency count trace)

                half_stim_index = ft.find_time_index(t, np.float(self.StimOnset + (self.StimEnd-self.StimOnset)/2))
                if AP_count > 5: # We only consider traces with more than 8.333 Hz = 5/600 ms spikes here
                    if np.sum(EphysObject._spikes_df['threshold_index'] < half_stim_index)!=0:
                        SFA = np.sum(EphysObject._spikes_df['threshold_index'] > half_stim_index) / \
                          np.sum(EphysObject._spikes_df['threshold_index'] < half_stim_index)
                    AHP_5 = EphysObject._spikes_df.loc[4, 'fast_trough_v'] - EphysObject._spikes_df.loc[4, 'threshold_v']
                    AP_threshold_5 = EphysObject._spikes_df.loc[4, 'threshold_v']
                    AP_amplitude_5 = EphysObject._spikes_df.loc[4, 'peak_height']
                    AP_width_5 = EphysObject._spikes_df.loc[4, 'width']*1000
                    UDR_5 = EphysObject._spikes_df.loc[4, 'upstroke_downstroke_ratio']

                # Add the (average) adaptation index
                AI = EphysObject._sweep_features['isi_adapt']
                AI_adapt_average = EphysObject._sweep_features['isi_adapt_average']

                # Add the latency
                latency = EphysObject._sweep_features['latency']*1000

                # Add the AP amp (average) adaptation (captures changes in AP amplitude during stimulation time)
                AP_amp_adapt = EphysObject._sweep_features['AP_amp_adapt']
                AP_amp_adapt_average = EphysObject._sweep_features['AP_amp_adapt_average']


                # Add the AP AHP, threshold, amplitude, width and UDR (upstroke-to-downstroke ratio) of the
                # first fired AP in the trace
                AHP = EphysObject._spikes_df.loc[0, 'fast_trough_v'] - EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_threshold = EphysObject._spikes_df.loc[0, 'threshold_v']
                AP_amplitude = EphysObject._spikes_df.loc[0, 'peak_height']
                AP_width = EphysObject._spikes_df.loc[0, 'width']*1000
                UDR = EphysObject._spikes_df.loc[0, 'upstroke_downstroke_ratio']

        # -------- #
        # 2nd part: features that derive standard stat moments, possibly good to perform inference
        std_pw = np.power(
            np.std(V[(t > self.StimOnset) & (t < self.StimEnd)]), np.linspace(3, n_mom, n_mom - 2)
        )
        std_pw = np.concatenate((np.ones(1), std_pw))
        moments = (
            spstats.moment(
                V[(t > self.StimOnset) & (t < self.StimEnd)], np.linspace(2, n_mom, n_mom - 1)
            )
            / std_pw
        )

        # resting potential and std
        rest_pot = np.mean(V[t < self.StimOnset])
        rest_pot_std = np.std(V[int(0.9 * self.StimOnset / dt) : int(self.StimOnset / dt)])


        # concatenation of summary statistics
        features = np.array(['AP threshold', 'AP amplitude', 'AP width', 'AHP', 'UDR', \
                '5th AP threshold', '5th AP amplitude', '5th AP width', '5th AHP', '5th UDR', \
                'AP count', 'AP amp adapt', \
                'AP average amp adapt', 'AP FF', 'AP CV', 'ISI adapt', 'ISI average adapt', 'ISI FF', 'ISI CV', \
                'latency', 'SFA', r'rest $V_{m}$ mean', r'rest $V_{m}$ std', \
                r'$V_{m}$ mean', r'$V_{m}$ std', r'$V_{m}$ skewness'])

        sum_stats_vec = np.concatenate(
            (
                np.array([AP_threshold, AP_amplitude, AP_width, AHP, UDR, \
                          AP_threshold_5, AP_amplitude_5, AP_width_5, AHP_5, UDR_5, \
                          AP_count, np.log(AP_amp_adapt), \
                          sigmoid(AP_amp_adapt_average, offset=1, steepness=50), \
                          np.log(AP_fano_factor), np.log(AP_cv), \
                          np.log(AI), sigmoid(AI_adapt_average, offset=1, steepness=2), \
                          np.log(fano_factor), np.log(cv), np.log(latency+0.1), SFA]),

                np.array(
                    [rest_pot, rest_pot_std, np.mean(V[(t > self.StimOnset) & (t < self.StimEnd)])]
                ),
                moments,
            )
        )
        # sum_stats_vec = sum_stats_vec[0:n_summary]
        if selected_stats != None:
            summary_stats = dict(zip(features[selected_stats],sum_stats_vec[selected_stats]))
        else:
            summary_stats = dict(zip(features,sum_stats_vec))
        return summary_stats

    def summarise(self, selected_stats=None, summary_func="v1"):
        """Calculate statistics that summarise the important features of trace object.
        Compared to "calculate_summary", the summary stats are ONLY recalculated IF
        self.Summary does not exist or the selected stats are different.

        Parameters
        ----------
        selected_stats : None, tuple, list or array
            Indices of summary stats to be used.
        summary_func : str
            A descriptive string like "v1" or "v2" can be provided to
            use one of the summary methods already implemented as part of Trace().

        Returns
        -------
        summary statistics : dict
            Contains key-value pairs of the summary statistics.
        """

        features = np.array([]) # features init

        if "1" in summary_func.lower():
            calculate_summary = self.calculate_summary_v1
            # features that are currently caculated
            features = np.array(['AP threshold', 'AP amplitude', 'AP width', 'AHP', 'UDR', \
                '5th AP threshold', '5th AP amplitude', '5th AP width', '5th AHP', '5th UDR', \
                'AP count', 'AP amp adapt', \
                'AP average amp adapt', 'AP FF', 'AP CV', 'ISI adapt', 'ISI average adapt', 'ISI FF', 'ISI CV', \
                'latency', 'SFA', r'rest $V_{m}$ mean', r'rest $V_{m}$ std', \
                r'$V_{m}$ mean', r'$V_{m}$ std', r'$V_{m}$ skewness'])
        if "2" in summary_func.lower():
            calculate_summary = self.calculate_summary_v2
            # features that are currently caculated
            features = np.array(['AP threshold', 'AP amplitude', 'AP width', 'AHP', \
                '3rd AP threshold', '3rd AP amplitude', '3rd AP width', '3rd AHP', \
                'AP count', 'AP amp adapt', \
                'AP average amp adapt', \
                'AP CV', 'ISI adapt', 'ISI CV', 'latency', 'SFA', \
                r'rest $V_{m}$ mean', r'$V_{m}$ mean', \
                r'$V_{m}$ std', r'$V_{m}$ skewness'])

        if self.Summary != {}:
            if set(self.Summary.keys()) == set(features[selected_stats].reshape(-1)):
                #print("The specified summary stats are the same as existing summary stats. Using existing summary instead.")
                #print("If you still want to recalculate the summary statistics, use calculate_summary instead.")
                return self.Summary
            else:
                summary_stats = calculate_summary(selected_stats=selected_stats)
        else:
            summary_stats = calculate_summary(selected_stats=selected_stats)

        self.Summary = summary_stats

        return self.Summary


class Cell:
    """Hold the parameters of the cell that is under investigation.

    Parameters
    ----------
    parameters : dict
        Contains as key the member variables of Cell and as values the parameter
        values of the Cell instance that is being created."""

    def __init__(self, parameters=None):
        self.C = 1e-2, # [F/m] usually 1 muF/cm^2
        self.RIn = None # [Ohm]
        self.Tau = None # [s]
        self.ASoma = None # [m^2]
        self.gLeak = None # [1/Ohm]
        self.Description = None
        self.Type = None

        if parameters != None:
            self.set_cell_attrs(parameters)

    def set_cell_attrs(self, parameters):
        for key in parameters.keys():
            self.__dict__[key] = parameters[key]


class Data(Cell, Trace):
    """Aggregate the collected data of electrophysiological measurements from current
    clamp expereminets and provides methods for easy import, preparation, analysis,
    visualisation and simulation/inference procedures.

    Parents
    -------
    Cell
    Trace """


    def __init__(self, observations=None, exp_params=None, cell_params=None):
        self.Observations = [] # imported traces
        self.Observations_selected = [] # traces to use
        self.NumObs = None
        self.ObsTime = None
        self.StimOnset = None
        self.StimDur = None
        self.StimEnd = None
        self.NumElec = None
        self.NumStimParadigms = None
        self.IStep = None
        self.I0Idx = None
        self.LiqJuncPotential = None
        self.dt = None
        self.Temp = None # [°C]

        self.Cell = Cell(cell_params)


        if type(observations) != type(None):
            self.import_observations(observations)
            self.Observations_selected = self.Observations.copy()

        self.set_attrs()
        self.set_exp_parameters(exp_params)


    def add(self, dct):
        """Add trace data to observations.

        Parameters
        ----------
        dct : dict
            Contains data needed to initialise trace object with Trace.__init__().
        """

        trace = Trace(dct)
        self.Observations.append(trace)

    def import_observations(self, observations):
        """Import several observation from .mat or dictionary.

        Parameters
        ----------
        observations : dict or str
            Path to .mat file that contains the data.
            Alternatively the dictionary containing the data, i.e. previously
            imported from a .mat file."""

        if type(observations) == str:
            self.from_mat(observations)

        if type(observations) == dict:
            self.from_dict(observations)

        if type(observations) == list or type(observations) == np.ndarray:
            if type(observations[0]) == type(Trace()):
                self.from_traces(observations)

    def __set_Iin_adjust_Vt(self, observations, I0Idx, IStep, NumStimParadigms):
        """Sets and adjusts both membrane voltages and injected currents for all
        observations. Takes into consideration the liquid junction potential,
        the stimulus paradigms and experimental sequence.

        Parameters
        ----------
        observations : dict or str
            Path to .mat file that contains the data.
            Alternatively the dictionary containing the data, i.e. previously
            imported from a .mat file.
        I0Idx : int
            Index that corresponds to where the current injected I = 0A.
        IStep : float
            Difference between successive current amplitudes.
        NumStimParadigms :
            The number of different currents that have been injected across all
            the observations.

        Returns
        -------
        observations : dict
            Observations, where all traces have been adjusted and a proper input
            current It has been added."""

        Iin_start = -self.I0Idx*self.IStep
        Iin_stop = Iin_start + (self.IStep*self.NumStimParadigms)
        Iin = np.arange(Iin_start, Iin_stop, self.IStep)

        # adjust trace parameters
        for trace, I in zip(observations,Iin):
            trace.IIn = I
            trace.Vt -= self.LiqJuncPotential
            trace.StimOnset = self.StimOnset
            trace.StimDur = self.StimDur
            trace.StimEnd = self.StimEnd
            #trace.It = np.zeros_like(trace.Vt)
            #stim_period = np.logical_and(trace.ts > self.StimOnset, trace.ts < self.StimEnd)
            #trace.It[stim_period] = I

        return observations

    def set_attrs(self):
        """Set member variables / trace parameters for all traces part of
        the observations."""
        for component in [self.Cell, self, self.Observations_selected[0]]:
            for key in component.__dict__.keys():
                if key in self.__dict__.keys():
                    self.__dict__[key] = component.__dict__[key]

    def set_exp_parameters(self, parameters):
        """Set member variables / experimental parameters.

        Parameters
        ----------
        parameters : dict
            Key-value pairs for member variables and there values."""
        if parameters != None:
            for key in parameters.keys():
                self.__dict__[key] = parameters[key]
                # if key in self.__dict__.keys(): # IS THIS EVEN NECCESARY ??
                #     self.__dict__[key] = parameters[key]

    def prepare(self):
        """Prepare the raw observational data. Adjust voltage traces and current
        input. Find the trace with I = 0A stimulation."""

        #first_obs = self.Observations_selected[0]
        #object_ephys = efex.EphysSweepFeatureExtractor(t = first_obs.ts, v = 1000*first_obs.Vt,
        #                                               start = self.StimOnset, end = self.StimEnd,
        #                                               filter = 10)
        #voltage_deflection_v, _ = object_ephys.voltage_deflection()
        #Vm = object_ephys._get_baseline_voltage()
        #V_defl = voltage_deflection_v

        #if  np.abs(Vm - V_defl) < 2:
        #    print("There appears not to be any hyperpolarisation,\
        #          you might have selected the wrong electrode for analysis.")
        #else:
        #    pass # possibly add functionality to choose different electrode

        self.NumStimParadigms = self.NumObs/self.NumElec

        Vts = [item.Vt for item in self.Observations_selected]
        self.I0Idx = np.argmin(list(map(np.var, Vts)))

        self.Observations_selected = self.__set_Iin_adjust_Vt(self.Observations_selected,
                                                 self.I0Idx, self.IStep, self.NumStimParadigms)


    def from_mat(self, PATH):
        """Import observations from .mat file.

        Parameters
        ----------
        PATH : str
            Path to .mat file."""

        dct = loadmat(PATH)
        self.from_dict(dct) # exp params need to be allowed

    def from_dict(self, data):
        """Import several observations from dictionary and set NumElec and NumObs
        accordingly.

        Parameters
        ----------
        data : dict
            Contains the all the obsered data in a specific format,
            i.e. as imported from a .mat file.
            {"Trace_X_X_X_X": [V(t_1),...V(t_N)], "Trace_X_X_X_Y": [V(t_1),...V(t_N)], ...}"""

        for key in data.keys():
            if "Trace" in key:
                trace_data = {"description": key, "data": data[key]}
                self.Observations.append(Trace(trace_data))

        self.NumElec = max([x.ElecIdx for x in self.Observations])
        self.NumObs = len(self.Observations)

    def from_traces(self, data):
        """Import several traces from a list or array of Trace objects.

        Parameters
        ----------
        data : list or numpy.ndarray
            Contains the all the list of trace objects."""

        self.Observations = data
        self.Observations_selected = data

    def select_subset(self, el_idxs=[2], t_cutoff=None, prepare=True):
        """Select subset of observed voltage traces by the electrode index,
        potenially trim the time axis and adjust the data according to the
        experimental parameters.

        Parameters
        ----------
        el_idxs : list
            List of electrode indexes for which to pick the data for analysis.
        t_cutoff : float
            Time point at which observed traces are cut off.
        prepare : bool
            Whether to prepare raw data for analysis and further calculations."""

        self.Observations_selected = []
        for trace in self.Observations.copy():
            if trace.ElecIdx in el_idxs:
                if t_cutoff != None:
                    trace.Vt = trace.Vt[trace.ts <= t_cutoff]
                    trace.ts = trace.ts[trace.ts <= t_cutoff]
                    trace.ObsTime = trace.ts[-1]
                    trace.NumBins = len(trace.ts)
                    if type(trace.It) != type(None):
                        trace.It = trace.It[trace.ts <= t_cutoff]

                self.Observations_selected.append(trace)
        if prepare:
            self.prepare()

    def at(self, IIn, nolistout = True):
        """Select voltage trace by the magnitude of injected current.

        Parameters
        ----------
        IIn : float or int
            Value of I in A or mA.
        nolistout : bool
            Whether to return output as list of traces or a single trace."""

        if type(IIn) == int or type(IIn) == float:
            if np.log10(abs(IIn)) > 0: # test if in mA or A
                IIn *= 1e-12
            sample = np.array([trace for trace in self.Observations_selected if abs(trace.IIn-IIn) < 1e-12])
            if nolistout:
                sample =  sample[0]
        else:
            if np.log10(abs(IIn[0])) > 0: # test if in mA or A
                IIn = np.array(IIn)*1e-12
            close_enough = lambda x, lst: (abs(np.array(lst)-x) < 1e-12).any() # closest I to input I
            sample = np.array([trace for trace in self.Observations_selected if close_enough(trace.IIn, IIn)])

        return sample


    def inspect(self, samplesize=10, random=False, at_idx=None, at_IIn=None, size=(12, 4), voltage_only=False, timewindow=None, savefig=False, fig_name="default"):
        """Plot voltage and possibly current trace.

        Parameters
        ----------
        samplesize : int
            Number of traces to plot.
        random : bool
            Whether to select traces at random or in order.
        at_idx : int or list
            One or multiple indices of voltage traces.
        at_IIn : int or list
            One or multiple input currents.
        size : tuple, list or ndarray
            Size of the plotted figure.
        timewindow : tuple, list or ndarray
            The voltage and current trace will only be plotted between (t1,t2).
            To be specified in secs.
        savefig : bool
            Determines whether or not to save the plotted figure.
        fig_name : str
            Integrates figure name into the filename of the saved figure."""

        # setting up axes
        axs = 0
        fig = plt.figure(1, figsize=size)
        if not voltage_only:
            gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1], figure=fig)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            axs = [ax1,ax2]
        else:
            axs = plt.subplot(111)

        if random:
            sample = np.random.choice(self.Observations_selected, size=samplesize)
        else:
            spacing = int(len(self.Observations_selected)/samplesize)
            sample = self.Observations_selected[::spacing]

        if at_idx != None:
            if type(at_idx) == int:
                sample = [self.Observations_selected[at_idx]]
            else:
                sample = [self.Observations_selected[idx] for idx in at_idx]

        if at_IIn != None:
            sample = self.at(at_IIn, nolistout = False)

        else:
            # common stats about the data
            obs = self.Observations_selected
            print("number of selected traces = {}".format(len(obs)))
            print("number of electrodes = {}".format(self.NumElec))
            print("number of points per trace = {}".format(obs[0].NumBins))
            print("total duration of recordings = {0:.0f}ms".format(obs[0].ts[-1]*1e3))
            print("dt = {0:.2f}ms; sampling frequency = {1:.0f}Hz".format(obs[0].dt*1e3, obs[0].SampFq))

        for trace in sample:
            try:
                label_str = "{0:.0f}pA".format(trace.IIn*1e12)
            except TypeError:
                label_str = ""
            trace.inspect(axes=axs,voltage_only=voltage_only, label=label_str, timewindow=timewindow)

        if savefig:
            plt.savefig('../data/figures/{}_8paramprior_support_vtraces_sample.png'.format(fig_name))

    def set_syn_current(self, R_in, tau):
        """Set synthetic current according to the fitted / estimated cell
        parameters. Calculate and set leak conductance gLeak and A_Soma.

        Parameters
        ----------
        R_in : int
            Input resistance. Can be obtained from I/V plot of the
            hyperpolarised traces.
        tau : bool
            The time constant, that can be obtained from the exponential decay
            of the hyperpolarised traces."""

        params = {"RIn": R_in, "gLeak": 1/R_in, "Tau": tau, "ASoma":tau/(R_in*self.Cell.C)}
        self.Cell.set_cell_attrs(params)

        for trace in self.Observations_selected:
            trace.It = np.zeros_like(trace.Vt)
            stim_period = np.logical_and(trace.ts > self.StimOnset, trace.ts < self.StimEnd)
            trace.It[stim_period] = trace.IIn/self.Cell.ASoma
            trace.ItA = trace.It*self.Cell.ASoma


    def fitcell2observations(self, t_probe=0.14, show=True, set_params=True, print_params=True):
        """Fit the cells parameters according to the observed voltage and current traces.

        Parameters
        ----------
        t_probe : float
            Time point at which to fit the I(V) curve that determines gLeak and R_in.
        show : bool
            Whether to display the fits and points chosen for fitting R_in.
        set_params : bool
            Whether to set the parameters according to the fitted values.
        print_params : bool
            Whether to print the fitted parameters."""

        observations = self.Observations_selected
        trace0 = observations[0]
        V_hyper = np.array([trace.Vt for trace in observations if trace.IIn < 0])
        ts = trace0.ts
        I_hyper = np.array([trace.IIn for trace in observations if trace.IIn < 0])

        # fit input resistance
        linear = lambda x,m,b: m*x+b
        m,b = curve_fit(linear, I_hyper/1e-12, V_hyper.T[abs(ts-t_probe) < self.dt/2.].reshape(-1)/1e-3)[0]

        # fit time constant
        taus = []
        for Vt, It in zip(V_hyper,I_hyper):
            EphysObject = efex.EphysSweepFeatureExtractor(ts, Vt, i = It,
                                                          start = trace0.StimOnset, end = trace0.StimEnd, filter = 10)
            taus.append(EphysObject.estimate_time_constant())
        Tau = np.mean(taus)

        if print_params:
            print("average tau = {:.3f} s".format(Tau))
            print("R_input = {:.3f}MOhm".format(m*1e3))


        if show:
            # plot V(t) and selected points V for fitting against I
            plt.figure()
            data, *_ = plt.plot(ts,V_hyper.T, c="grey", label="data")
            fit, *_ = plt.plot(np.ones(10)*t_probe,V_hyper.T[abs(ts-t_probe) < self.dt/2].reshape(-1),
                               "x", c="black", label="fit")
            plt.ylabel("V")
            plt.xlabel("t")
            plt.legend([data,fit], ["data", "fit"])

            # plot V(I) and linear fit
            plt.figure()
            plt.plot(I_hyper, V_hyper.T[abs(ts-t_probe) < self.dt/2].reshape(-1),
                     "x", c="black", label="data")
            plt.plot(I_hyper, linear(I_hyper,m/1e-9,b/1e3), c="red", label="fit")
            plt.ylabel("V")
            plt.xlabel("I")
            plt.legend()
            plt.show()

        if set_params:
            self.set_syn_current(m*1e9, Tau)
