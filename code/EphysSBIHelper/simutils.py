import numpy as np
import torch

# parallel processing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# HH models
from EphysSBIHelper.simulators.HH_Brian2 import HH_Br2
from EphysSBIHelper.simulators import HHsimulatorWrapper
from EphysSBIHelper.simulators.HH_python import HHsimulator
from EphysSBIHelper.datautils import Trace

def prepare_HH_input(x):
    """Takes input vector to the HH simulator and
    tests whether it conforms with the neccessary
    shape. If not, it will be reshaped accordingly
    so it possess a shape of of (1,-1).

    Parameters
    ----------
    x : numpy.ndarray or torch.tensor or list
        Inputparameters to the HH.

    Returns
    -------
    x : torch.tensor
        Appropriately reshaped input to the HH."""

    if type(x) == np.ndarray:
        x = torch.tensor(x)

    num_dim = x.dim()

    if num_dim == 1:
        x = x.reshape(1,-1)
    return x

def runHH(model_params, trace=None, syn_current=None, using="C++", mode="auto"):
    """Run the Hodgkin Huxley model with provided parameters and stimulation
    protocol and simulate the behaviour of the membrane voltage.
    Runs C code in the backend or via Brian2.

    Parameters
    ----------
    model_params : ndarray
        .Parameter vector that contains the HH model parameters.
    trace : Trace
        Trace object for the observed voltage trace, containing all relevant values
        of the stimulation protocol.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    using : str
        Specify "Brian2", "C++" or python implementaion of the HH model.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace"). If ("auto") is selected, parameter models
        with > 8 parameters will be considered to have ASoma and V0 included,
        -> ("model_params") will thus be selected. For models with 8 params,
        ("trace") will be used.

    Returns
    -------
    simulated_trace : Trace
        Trace object that contains the simulated voltage trace as well as all
        other relevant parameters of the stimulation protocol."""

    # initialise some simulation parameters
    I = 0
    ts = 0
    dt = 0
    V0 = 0
    A = 0

    model_params = prepare_HH_input(model_params)
    if "b" not in using.lower() and model_params.shape[0] > 1:
        print("Too many parameters provided at once. Switching to Brian2 model.")
        print("To use large batches of parameters, use *_batches() functions.")
        using = "Brian2"

    # depending on the type of input, the simulation parameters are extracted
    if trace != None:
        syn_current = trace.get_syn_current()

    ts = syn_current["ts"]*1e3 # convert from s to ms
    dt = syn_current["dt"]*1e3 # convert from s to ms
    stim_start = syn_current["StimOnset"]*1e3 # convert from s to ms
    stim_end = syn_current["StimEnd"]*1e3 # convert from s to ms
    stim_dur = syn_current["StimDur"]*1e3 # convert from s to ms
    IA = syn_current["ItA"]*1e12 # A to pA

    if "auto" in mode.lower():
        if model_params.shape[1] > 8:
            mode = "model_params"
        else:
            mode = "trace"
    if "trace" in mode.lower():
        I = syn_current["It"].reshape(1,-1)*1e2 # convert from A/m^2 to muA/cm^2
        V0 = syn_current["V0"]*1e3 # convert from V to mV
        A = (IA[IA != 0] / I[I != 0])[0]*1e2 #???
    elif "params" in mode.lower():
        V0 = model_params[:,8]*1e3
        A = model_params[:,9]*1e12
        I = np.array(syn_current["ItA"]/A.reshape(-1,1)*1e14)
        model_params = model_params[:,:-2]
    else:
        raise ValueError("currently only 'auto','model_params' and 'trace' mode are supported.")

    t_fin = ts[-1]

    # run the HH model in C
    if "c" in using.lower():
        V_out = HHsimulatorWrapper.runHH(list(model_params.numpy().reshape(-1)), 1, float(V0), list(I.reshape(-1)), dt, t_fin)
        V_out = torch.tensor(V_out)
        Vs = V_out.reshape(1,-1)

    # run the HH model in Brian2
    if "b" in using.lower():
        ts, Vs, It = HH_Br2(model_params, np.array(V0).reshape(1,-1), np.array(A).reshape(1,-1), np.array(IA), dt, t_fin)
        ts += 5e-03  # only for Br2

    # run in python
    if "p" in using.lower():
        Vs = HHsimulator(model_params, 1, float(V0), np.array(I).reshape(-1), dt, t_fin)
        Vs = Vs.reshape(1, -1)

    # set up Trace object
    if Vs.shape[0] == 1:
        trace_params = {"dt": dt*1e-3, "It": I.reshape(-1)*1e-2, "ItA": IA*1e-12, "StimOnset": stim_start*1e-3, "StimEnd": stim_end*1e-3, "StimDur": stim_dur*1e-3}
        trace_init = {"data": np.vstack([ts*1e-3,Vs.reshape(-1)*1e-3]).T, "params": trace_params}
        simulated_trace = Trace(trace_init)
        return simulated_trace
    else:
        # Brian2 implementation can ran multiple simulations with one call
        simulated_traces = []
        for Its, Vts in zip(It, Vs):
            trace_params = {"dt": dt*1e-3, "It": np.array((It[0]/A.reshape(-1,1)).reshape(-1))*1e-2, "ItA": IA*1e-12, "StimOnset": stim_start*1e-3, "StimEnd": stim_end*1e-3, "StimDur": stim_dur*1e-3}
            trace_init = {"data": np.vstack([ts*1e-3,Vts.reshape(-1)*1e-3]).T, "params": trace_params}
            simulated_trace = Trace(trace_init)
            simulated_traces.append(simulated_trace)
        return simulated_traces

def simulate_and_summarise(params, trace=None, syn_current=None, selected_stats=None, using="C++", mode="auto", summary_func="v1"):
    """Run the Hodgkin Huxley model with provided parameters and stimulation
    protocol, simulates and summarises the behaviour of the membrane voltage.

    Parameters
    ----------
    model_params : ndarray
        The parameter inputs to the HH model [r'$g_{Na}$', r'$g_{K}$',r'$g_{M}$',
        r'$g_{leak}$', r'$\tau_{max}$', r'$V_{T}$', r'$E_{leak}$', 'time constant factor'].
    trace : Trace
        Trace object for the observed voltage trace, containing all relevant values
        of the stimulation protocol.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    selected_stats : None or list
        Selected summary statistics to output, by their index.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace").
    summary_func : str
        A descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().

    Returns
    -------
    summary_tensor : torch.tensor
        Tensor object that contains the summary statistics of the simulated
        voltage trace."""

    result = runHH(params, trace, syn_current, using, mode)
    result.summarise(selected_stats, summary_func=summary_func)
    summary_as_lst = list(result.Summary.values())
    summary_tensor = torch.as_tensor(summary_as_lst)
    return summary_tensor

def simulate_and_summarise_wrapper(params, syn_current, selected_stats=None, using="C++", mode="auto", summary_func="v1", output2file=False, file=None):
    """Runs the HH simulator for a set of parameters and stimulation current.
    Calculates a selection of summary statistics for the simulated trace.

    Parameters
    ----------
    params : ndarray
        The parameter inputs to the HH model [r'$g_{Na}$', r'$g_{K}$',r'$g_{M}$',
        r'$g_{leak}$', r'$\tau_{max}$', r'$V_{T}$', r'$E_{leak}$', 'time constant factor'].
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    selected_stats : ndarray, list
        Lists of indices corresponding to summary statistics to return, as compared to self.Summary
        with input == None.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace").
    sumary_func : function or str
        A function that takes voltage trace V(t), current trace I(t), time
        points t and time step dt and outputs a dictionary with summary stats
        and descriptions.
        Alternatively a descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().
    output2file : bool
        Determines whether to redirect summary results to file for safekeeping,
        instead of outputting them.
    file : file
        An open file object.

    Returns
    -------
    summstats_vec : ndarray
        Vector containing the results of the calculated summary statistics."""

    summstats_vec=[]
    if "b" in using.lower():
        traces = runHH(params, syn_current=syn_current, using=using, mode=mode)
        if type(traces) != list:
            traces = [traces]
        for trace in traces:
            summary = trace.summarise(selected_stats, summary_func=summary_func)
            summary_as_lst = list(summary.values())
            summary_tensor = torch.as_tensor(summary_as_lst)
            summstats_vec.append(summary_tensor)
    else:
        for param in params:
            summary = simulate_and_summarise(param, syn_current=syn_current, selected_stats=selected_stats, using=using, mode=mode, summary_func=summary_func)
            summstats_vec.append(summary)

    if output2file:
        save_preliminary_results(params, summstats_vec, file)
        return None
    else:
        return summstats_vec

def simulate_and_summarise_batches(theta, syn_current, num_workers=1, batch_size=25, selected_stats=None, using="C++", mode="auto", summary_func="v1", output2file=False, filename=None, header=None):
    """Runs the HH simulator and summarieses the results for a stimulation current and
    a large sample of parameter sets. Computations can be done in batches and in parrallel.

    Parameters
    ----------
    theta : ndarray, Tensor
        The parameter inputs to the HH model [r'$g_{Na}$', r'$g_{K}$',r'$g_{M}$',
        r'$g_{leak}$', r'$\tau_{max}$', r'$V_{T}$', r'$E_{leak}$', 'time constant factor'].
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    num_workers : int
        Number of processes/threads used to simulate and summarise the specified traces.
    batch_size : int
        Number of parameters to be processed per batch.
    selected_stats : ndarray, list
        Lists of indices corresponding to summary statistics to return, as compared to self.Summary
        with input  == None.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace").
    sumary_func : function or str
        A function that takes voltage trace V(t), current trace I(t), time
        points t and time step dt and outputs a dictionary with summary stats
        and descriptions.
        Alternatively a descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().
    output2file : bool
        Determines whether to redirect summary results to file for safekeeping,
        instead of outputting them.
    filename : str
        Name of the outputfile. If not specified, a default naming scheme will be chosen.
    header : str
        First row of the output file. E.g. Column names.

    Returns
    -------
    stats : ndarray
        Vector containing the calculated summary statistics for all sets of parameters."""

    # Strangely pickling of Trace does not work if it is not imported from a different Module!!!
    # parallelisation adpated from code courtesy of @ybernaerts.

    if output2file:
        file = file_init(fname=filename, header=header, theta=theta)
    else:
        file = None

    theta = prepare_HH_input(theta)
    batches = torch.split(theta, batch_size, dim=0)
    num_sims = theta.shape[0]

    summaries = Parallel(n_jobs=num_workers)(
                        delayed(simulate_and_summarise_wrapper)(batch, syn_current, selected_stats, using, mode, summary_func, output2file, file)
                        for batch in tqdm(
                            batches,
                            disable=False,
                            desc=f"Simulating and summarising {num_sims} results in {len(batches)} batches.",
                            total=len(batches),
                        )
                    )
    # recombining and reshaping results from the different processing threads

    if not output2file:
        stats = torch.stack(summaries[0])
        for i in range(len(summaries)-1):
            stats=torch.cat((stats, torch.stack(summaries[i+1])))

        return stats
    else:
        file.close()
        return None

def runHH_wrapper(params, trace_obs=None, syn_current=None, using="C++", mode="auto"):
    """Wrapper function, that allows for the input of a batch of parameters.
    Neccessary for multithreading.

    Parameters
    ----------
    params : ndarray
        The parameter inputs to the HH model [r'$g_{Na}$', r'$g_{K}$',r'$g_{M}$',
        r'$g_{leak}$', r'$\tau_{max}$', r'$V_{T}$', r'$E_{leak}$', 'time constant factor'].
    trace_obs : Trace
        Trace object for the observed voltage trace.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace").

    Returns
    -------
    traces : list[Trace]
        List containing the simulation results in the form of Trace() objects for
        the batch of input parameters."""

    if "b" in using.lower():
        traces = runHH(params, trace_obs, syn_current, using, mode)
    else:
        traces=[]
        for param in params:
            trace = runHH(param, trace_obs, syn_current, using, mode)
            traces.append(trace)
    return traces

def summary_wrapper(traces, selected_stats=None, summary_func=None):
    """Wrapper function that applies the Trace.summary() method
    to a batch of simulated traces. Neccessary for multithreading.

    Parameters
    ----------
    traces : list[Trace]
        List containing Trace() objects that need to be summarised.
   selected_stats : ndarray, list
        Lists of indices corresponding to summary statistics to return, as compared to self.Summary
        with input  == None.
    sumary_func : function or str
        A function that takes voltage trace V(t), current trace I(t), time
        points t and time step dt and outputs a dictionary with summary stats
        and descriptions.
        Alternatively a descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().

    Returns
    -------
    summary_stats : list
        List containing the summary tensors for the batch of simulations."""

    summary_stats = []
    for trace in traces:
        trace.summarise(selected_stats, summary_func=summary_func)
        summary_as_lst = list(trace.Summary.values())
        summary_tensor = torch.as_tensor(summary_as_lst)
        summary_stats.append(summary_tensor)

    return summary_stats

def summarise_batches(traces, num_workers=1, batch_size=25, selected_stats=None, summary_func=None):
    """Summarises the features for a large collection of traces.
    Computations can be done in batches and in parrallel.

    Parameters
    ----------
    traces : numpy.array[Trace] or list(Trace)
        List containing Trace() objects that need to be summarised.
    num_workers : int
        Number of processes/threads used to simulate and summarise the specified traces.
    batch_size : int
        Number of parameters to be processed per batch.
    selected_stats : ndarray, list
        Lists of indices corresponding to summary statistics to return, as compared to self.Summary
        with input  == None.
    sumary_func : function or str
        A function that takes voltage trace V(t), current trace I(t), time
        points t and time step dt and outputs a dictionary with summary stats
        and descriptions.
        Alternatively a descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().

    Returns
    -------
    stats : ndarray
        Vector containing the calculated summary statistics for all sets of parameters of all batches."""

    # Strangely pickling of Trace does not work if it is not imported from a different Module!!!
    traces = np.array(traces)
    num_sims = traces.shape[0]

    batches = np.split(traces,int(num_sims/batch_size))
    summaries = Parallel(n_jobs=num_workers)(
                        delayed(summary_wrapper)(batch, selected_stats, summary_func)
                        for batch in tqdm(
                            batches,
                            disable=False,
                            desc=f"Summarising {num_sims} results in {len(batches)} batches.",
                            total=len(batches),
                        )
                    )
    stats = torch.stack(summaries[0])
    for i in range(len(summaries)-1):
        stats=torch.cat((stats, torch.stack(summaries[i+1])))
    return stats

def simulate_batches(theta, trace_obs=None, syn_current=None, num_workers=1, batch_size=25, using="C++", mode="auto"):
    """Runs the HH simulator for a stimulation current and a large sample of parameter sets.
    Computations can be done in batches and in parrallel.

    Parameters
    ----------
    theta : ndarray, Tensor
        The parameter inputs to the HH model [r'$g_{Na}$', r'$g_{K}$',r'$g_{M}$',
        r'$g_{leak}$', r'$\tau_{max}$', r'$V_{T}$', r'$E_{leak}$', 'time constant factor'].
    trace_obs : Trace
        Trace object for the observed voltage trace.
    syn_current : dict
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    num_workers : int
        Number of processes/threads used to simulate and summarise the specified traces.
    batch_size : int
        Number of parameters to be processed per batch.
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace").

    Returns
    -------
    results : ndarray
        Vector containing the simulated traces for all input sets of parameters."""

    theta = prepare_HH_input(theta)
    batches = torch.split(theta, batch_size, dim=0)
    num_sims = theta.shape[0]

    # parallelisation adpated from code courtesy of @ybernaerts.
    results = Parallel(n_jobs=num_workers)(
                    delayed(runHH_wrapper)(batch, trace_obs, syn_current, using, mode)
                    for batch in tqdm(
                        batches,
                        disable=False,
                        desc=f"Running {num_sims} simulations in {len(batches)} batches.",
                        total=len(batches),
                    )
                )

    flat_list = []
    for item in results:
        if type(item) == list:
            flat_list += item
        else:
            flat_list.append(item)

    results = np.array(flat_list).reshape(-1)
    return results

def file_init(fname, header, theta, path="../data/"):
    """Helper function to initialise file to
    store results of simulations.

    Parameters
    ----------
    fname : str
        Specifies the file name.
    header : str
        Specifies the first / header row of the file.
    theta : torch.tensor numpy.ndarray
        Simulation parameters. Used for default file naming.
    path : str
        Specifies Path for the storage of the resulting file.

    Returns : file
        An open file object in append binary ("ab") mode."""

    if fname == None:
        rnd_str = "".join(np.random.choice(list("abcdefghajkalmop")) for i in range(5))
        fname = path + "{}_{}params_{}.dat".format(theta.shape[0], theta.shape[1], rnd_str)
    if header != None:
        with open(fname, "a") as f:
            if f.read(1) != "":
                f.write(f,header)
    file = open(fname, "ab")
    return file

def save_preliminary_results(thetas, batch_results, file):
    """Concatenates parameters and summary results and appends it
    to an open file.

    Parameters
    ----------
    thetas : torch.tensor
        Contains the simulation parameters.
    batch_results : list(torch.tensor)
        List that contains summary results of current batch.
    file : file
        An open file object in append and binary mode ("ab").
    """
    results = torch.hstack([thetas, torch.vstack(batch_results)]).numpy()
    np.savetxt(file,results)
