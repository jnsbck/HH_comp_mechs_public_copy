import numpy as np

def HHsimulator(model_params, seed, V0, I, dt, t_fin):
    """Adpated from code courtesy of @ybernaerts."""
    t_start = 0
    t_dur = t_fin - t_start
    size = int(t_dur/dt)+1
    model_params = np.array(model_params).reshape(-1)
    # parameters to be inferred

    # Maximum sodium concuctance
    gbar_Na = model_params[0] # mS/cm2
    gbar_Na.astype(float)

    # Maximum potassium concuctance
    gbar_K = model_params[1] # mS/cm2
    gbar_K.astype(float)

    # Maximum conductance for adaptive potassium currents (these can induce firing rate adaptations)
    gbar_M = model_params[2] # mS/cm2
    gbar_M.astype(float)

    # Leak conductance
    g_leak = model_params[3] # mS/cm2
    g_leak.astype(float)

    # time constant of the adaptive potassium current (gives an idea of when these channels will turn active)
    tau_max = model_params[4] # ms
    tau_max.astype(float)

    # Vt: a 'threshold' voltage that can influence the dynamics of all channels
    Vt = model_params[5] # mV
    Vt.astype(float)

    # Leak potential
    E_leak = model_params[6] # mV
    E_leak.astype(float)

    # A factor that can change the amount of injected current, or equivalently change the compartment area initially
    # deduced if it sees fit to do so
    #I_scale = model_params[0,7]
    #I_scale.astype(float)

    # A factor that can make the dynamics of Na+ and K+ channels faster/slower. This was included to find simulations
    # that can recover the shape of the 1st action potential fired.
    rate_to_SS_factor = model_params[7]
    rate_to_SS_factor.astype(float)

    # fixed parameters (g_leak, C from fitting hyperpolarization trace)
    #g_leak = 0.117  # mS/cm2
    #Vt = -60.0  # mV
    #E_leak = np.mean(voltage_obs[0:2500, curr_index])  # mV
    nois_fact = 0.1  # uA/cm2
    C = 1  # uF/cm2
    E_Na = 53  # mV            # TODO: check with Federico
    E_K = -90  # mV            # TODO: check with Federico
    Q10=3
    T_1 = 36                           # °C, from paper Martin Pospischil et al.
    T_2 = 34                           # °C, experiment was actually done at 34 °C
    T_adj_factor = Q10**((T_2-T_1)/10) # temperature coeff., https://en.wikipedia.org/wiki/Q10_(temperature_coefficient)

    tstep = float(dt)

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    ####################################
    # kinetics
    def efun(z):
        if np.abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (np.exp(z) - 1)

    def alpha_m(x):
        v1 = x - Vt - 13.0
        return 0.32 * efun(-0.25 * v1) / 0.25

    def beta_m(x):
        v1 = x - Vt - 40
        return 0.28 * efun(0.2 * v1) / 0.2

    def alpha_h(x):
        v1 = x - Vt - 17.0
        return 0.128 * np.exp(-v1 / 18.0)

    def beta_h(x):
        v1 = x - Vt - 40.0
        return 4.0 / (1 + np.exp(-0.2 * v1))

    def alpha_n(x):
        v1 = x - Vt - 15
        return 0.032 * efun(-0.2 * v1) / 0.2

    def beta_n(x):
        v1 = x - Vt - 10.0
        return 0.5 * np.exp(-v1 / 40)

    # steady-states and time constants
    def tau_n(x):
        return rate_to_SS_factor*1 / (alpha_n(x) + beta_n(x))

    def n_inf(x):
        return alpha_n(x) / (alpha_n(x) + beta_n(x))

    def tau_m(x):
        return rate_to_SS_factor*1 / (alpha_m(x) + beta_m(x))

    def m_inf(x):
        return alpha_m(x) / (alpha_m(x) + beta_m(x))

    def tau_h(x):
        return rate_to_SS_factor*1 / (alpha_h(x) + beta_h(x))

    def h_inf(x):
        return alpha_h(x) / (alpha_h(x) + beta_h(x))

    # slow non-inactivating K+
    def p_inf(x):
        v1 = x + 35.0
        return 1.0 / (1.0 + np.exp(-0.1 * v1))

    def tau_p(x):
        v1 = x + 35.0
        return tau_max / (3.3 * np.exp(0.05 * v1) + np.exp(-0.05 * v1))

    ####################################
    # simulation from initial point
    V = np.zeros(size)  # voltage
    n = np.zeros(size)
    m = np.zeros(size)
    h = np.zeros(size)
    p = np.zeros(size)

    V[0] = float(V0)
    n[0] = n_inf(V[0])
    m[0] = m_inf(V[0])
    h[0] = h_inf(V[0])
    p[0] = p_inf(V[0])

    for i in range(1, size):
        tau_V_inv = (
            (m[i - 1] ** 3) * gbar_Na * h[i - 1]
            + (n[i - 1] ** 4) * gbar_K
            + g_leak
            + gbar_M * p[i - 1]
        ) / C
        V_inf = (
            (m[i - 1] ** 3) * gbar_Na * h[i - 1] * E_Na
            + (n[i - 1] ** 4) * gbar_K * E_K
            + g_leak * E_leak
            + gbar_M * p[i - 1] * E_K
            + I[i - 1]
            + nois_fact * rng.randn() / (tstep ** 0.5)
        ) / (tau_V_inv * C)
        V[i] = V_inf + (V[i - 1] - V_inf) * np.exp(-tstep * tau_V_inv)
        n[i] = n_inf(V[i]) + (n[i - 1] - n_inf(V[i])) * np.exp((-tstep*T_adj_factor / tau_n(V[i])))
        m[i] = m_inf(V[i]) + (m[i - 1] - m_inf(V[i])) * np.exp((-tstep*T_adj_factor / tau_m(V[i])))
        h[i] = h_inf(V[i]) + (h[i - 1] - h_inf(V[i])) * np.exp((-tstep*T_adj_factor / tau_h(V[i])))
        p[i] = p_inf(V[i]) + (p[i - 1] - p_inf(V[i])) * np.exp((-tstep*T_adj_factor / tau_p(V[i])))

        Vs = np.array(V)
    return Vs
