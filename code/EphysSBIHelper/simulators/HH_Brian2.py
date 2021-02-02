import numpy as np

# Brian2
import brian2 as br2
br2.prefs.codegen.target = 'cython' # for C compile. Alternatively use "numpy"


def HH_Br2(model_params, V0, A, ItA, dt, t_fin):
    """Run the Hodgkin Huxley model with provided parameters and stimulation
    protocol. Brian2 Implementation.

    Adpated from code courtesy of @ybernaerts.

    Parameters
    ----------
    model_params : ndarray
        .Parameter vector that contains the HH model parameters.

    Returns
    -------
    t, Vm, I_inj : np.ndarray
        Timepoints, membrane voltage and the injected current are returned."""

    model_params = np.asarray(model_params, float)
    N = model_params.shape[0]

    ####################
    # Setting up the injection current, the model equations, further initialisations and run the model ;)

    I = br2.TimedArray(np.vstack( [[ItA]*N] ).T, dt = dt*br2.msecond)

    # The conductance-based model
    eqs = '''
         dVm/dt = -(gNa*m**3.0*h*(Vm - ENa) + gK*n**4.0*(Vm - EK) + gleak*(Vm - El) - I_inj +
                   (gM*p*(Vm - EK))) / C : volt

         I_inj = I(t, i)*pA : amp

         dm/dt = (alpham*(1.0-m) - betam*m) * t_adj_factor / rate_to_SS_factor : 1
         dn/dt = (alphan*(1.0-n) - betan*n) * t_adj_factor / rate_to_SS_factor : 1
         dh/dt = (alphah*(1.0-h) - betah*h) * t_adj_factor / rate_to_SS_factor : 1
         dp/dt = ((p_inf - p)/tau_p) * t_adj_factor : 1

         alpham = (-0.32/mV) * (Vm - VT - 13.0*mV) / (exp((-(Vm - VT - 13.0*mV))/(4.0*mV)) - 1.0)/ms : Hz
         betam = (0.28/mV) * (Vm - VT - 40.0*mV) / (exp((Vm - VT - 40.0*mV)/(5.0*mV)) - 1.0)/ms : Hz

         alphah = 0.128 * exp(-(Vm - VT - 17.0*mV) / (18.0*mV))/ms : Hz
         betah = 4.0/(1.0 + exp((-(Vm - VT - 40.0*mV)) / (5.0*mV)))/ms : Hz

         alphan = (-0.032/mV) * (Vm - VT - 15.0*mV) / (exp((-(Vm - VT - 15.0*mV)) / (5.0*mV)) - 1.0)/ms : Hz
         betan = 0.5*exp(-(Vm - VT - 10*mV) / (40*mV))/ms : Hz

         p_inf = 1.0/(1.0 + exp(-(Vm + 35.0*mV)/(10.0*mV))) : 1
         tau_p = (tau_max/1000.0)/(3.3 * exp((Vm + 35.0*mV)/(20.0*mV)) + exp(-(Vm + 35.0*mV)/(20.0*mV))) : second

         gNa : siemens
         gK : siemens
         gleak : siemens
         gM : siemens
         tau_max : second
         VT : volt
         El : volt
         rate_to_SS_factor : 1
         C : farad

         ENa : volt (shared)
         EK : volt (shared)

         t_adj_factor : 1 (shared)
         '''
    neurons = br2.NeuronGroup(N, eqs, method='exponential_euler', name='neurons', dt=dt*br2.msecond)

    ###################
    # Model parameter initialisations
    # Some are set, some are performed inference on
    # Inspired by Martin Pospischil et al. "Minimal Hodgkin-Huxley type models for
    # different classes of cortical and thalamaic neurons".
    area = A*br2.umetre**2  # um2

    neurons.gNa = model_params[:,0]*br2.mS/br2.cm**2*area
    neurons.gK = model_params[:,1]*br2.mS/br2.cm**2*area
    neurons.gM = model_params[:,2]*br2.mS/br2.cm**2*area
    neurons.gleak = model_params[:,3]*br2.mS/br2.cm**2*area
    neurons.tau_max = model_params[:,4]*br2.second
    neurons.VT = model_params[:,5]*br2.mV
    neurons.El = model_params[:,6]*br2.mV
    neurons.C = 1.0*br2.uF/br2.cm**2*area
    neurons.rate_to_SS_factor = model_params[:,7]

    neurons.Vm = V0*br2.mV #V0

    neurons.ENa = 53.0*br2.mV
    neurons.EK = -90.0*br2.mV

    # It is important to adapt your kinetics to the temperature of your experiment
    # temperature coeff., https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)
    T_1 = 36.0        # °C, from paper MartinPospischil et al.
    T_2 = 34.0        # °C, experiment was actually done at 25 °C
    Q10 = 3.0           # temperature coeff.

    neurons.t_adj_factor = Q10**((T_2 - T_1)/10.0)

    # set monitoring
    Vm_mon = br2.StateMonitor(neurons, ['Vm', 'I_inj'], record = True, name = 'Vm_mon') # Specify what to record


    # init
    neurons.m = '1/(1 + betam/alpham)'         # Would be the solution when dm/dt = 0
    neurons.h = '1/(1 + betah/alphah)'         # Would be the solution when dh/dt = 0
    neurons.n = '1/(1 + betan/alphan)'         # Would be the solution when dn/dt = 0
    neurons.p = 'p_inf'                        # Would be the solution when dp/dt = 0

    # run simulation
    br2.run(t_fin*br2.ms)

    return Vm_mon.t/br2.ms, Vm_mon.Vm/br2.mV, Vm_mon.I_inj/br2.pA
