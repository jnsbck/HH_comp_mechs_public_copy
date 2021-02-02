import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from EphysSBIHelper.simutils import runHH
from copy import deepcopy

class MDNPosterior(DirectPosterior):
    """Wrapperclass for DirectPosterior objects that were trained using
    a Mixture Density Network (MDN). Replaces sample and log_prob functions.
    Enables the extraction of the Mixture of Gaussian parameters.

    Parameters
    ----------
    MDN_Posterior : DirectPosterior
        DirectPosterior object, output of inference.build_posterior(density_estimator),
        that was trained with a MDN."""

    def __init__(self, MDN_Posterior):
        if "MultivariateGaussianMDN" in MDN_Posterior.net.__str__():
            # wrap copy of input object into self
            self.__class__ = type("MDNPosterior",
                                  (self.__class__, deepcopy(MDN_Posterior).__class__),
                                  {})
            self.__dict__ = deepcopy(MDN_Posterior).__dict__

            # MoG parameters
            self.S = None
            self.m = None
            self.mc = None
            self.support = self._prior.support

            self.extract_mixture_components()

        else:
            raise AttributeError("Posterior does not contain a MDN.")

    def check_support(self, X):
        """Takes a set of points X with X.shape[0] being the number of points
        and X.shape[1] the dimensionality of the points and checks, each point
        for its prior support.

        Parameters
        ----------
        X : torch.tensor
            Contains a set of multidimensional points to check
            against the prior support of the posterior object.

        Returns
        -------
        within_support : torch.array[bool]
            Boolean array representing, whether a sample is within the
            prior support or not.
        """

        lbound = self.support.lower_bound
        ubound = self.support.upper_bound

        within_support = torch.logical_and(lbound < X, X < ubound)

        return torch.all(within_support, dim=1)

    def extract_mixture_components(self, x=None):
        """Extracts the Mixture of Gaussians (MoG) parameters
        from the MDN at either the default x or input x.

        Adpated from code courtesy of @ybernaerts.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray
            x at which to evaluate the MDN in order
            to extract the MoG parameters.
        """
        if x == None:
            encoded_x = self.net._embedding_net(self.default_x)
        else:
            encoded_x = self.net._embedding_net(torch.tensor(x, dtype=torch.float32))
        dist = self.net._distribution
        logits, m, prec, *_ = dist.get_mixture_components(encoded_x)
        norm_logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        scale = self.net._transform._transforms[0]._scale
        shift = self.net._transform._transforms[0]._shift


        self.mc = torch.exp(norm_logits).detach().double()
        self.m = ((m-shift)/scale).detach()[0].double()


        L = np.linalg.cholesky(prec[0].detach())
        C = torch.tensor(np.linalg.inv(L))
        self.S = (C.transpose(2,1)@C)
        A_inv = torch.inverse(scale*np.eye(self.S.shape[1]))
        self.S = A_inv@self.S.double()@A_inv.T


        return self.mc, self.m, self.S

    def log_prob(self, X, individual=False):
        """Evaluates the Mixture of Gaussian (MoG)
        probability density function at a value x.

        Adpated from code courtesy of @ybernaerts.

        Parameters
        ----------
        X : torch.Tensor or numpy.ndarray
            Values at which to evaluate the MoG pdf.
        individual : bool
            If True the probability density is returned for each cluster component.

        Returns
        -------
        log_prob : numpy.ndarray
            Log probabilities at values specified by X.
        """

        pdf = torch.zeros((X.shape[0], self.m.shape[0]))
        for i in range(self.m.shape[0]):
            pdf[:,i] = mulnormpdf(X, self.m[i], self.S[i]) * self.mc[0,i]
        if individual:
            return torch.log(pdf)
        else:
            log_factor = torch.log(self.leakage_correction(x=self.default_x))
            return torch.log(torch.sum(pdf, axis=1))-log_factor

    def sample(self, num_samples=(1,)):
        """Draw samples from a Mixture of Gaussians (MoG)

        Adpated from code courtesy of @ybernaerts.
        Parameters
        ----------
        num_samples : tuple
            The number of samples to draw from the MoG distribution.

        Returns
        -------
        X : numpy.ndarray
             A matrix with samples rows, and input dimension columns.
        """

        K, D = self.m.shape # Determine dimensionality

        # Cluster selection
        cs_mc = np.cumsum(self.mc)
        cs_mc = np.concatenate(([0], cs_mc))
        sel_idx = np.random.rand(num_samples[0])

        # Draw samples
        res = np.zeros((num_samples[0], D))
        for k in range(K):
            idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
            ksamples = np.sum(idx)

            # draw initial samples
            drawn_samples = np.random.multivariate_normal(\
                self.m[k], self.S[k], ksamples)

            # check if samples are within the support and how many are not
            supported = self.check_support(torch.tensor(drawn_samples))
            num_not_supported = torch.count_nonzero(~supported)
            drawn_samples_in_support = drawn_samples[np.array(supported)]
            if num_not_supported > 0:
                # resample until all samples are within the prior support
                while num_not_supported > 0:
                    # resample
                    redrawn_samples = np.random.multivariate_normal(\
                    self.m[k], self.S[k], int(num_not_supported))

                    # reevaluate support
                    supported = self.check_support(torch.tensor(redrawn_samples))
                    num_not_supported = torch.count_nonzero(~supported)
                    redrawn_samples_in_support = redrawn_samples[np.array(supported)]
                    # append the samples
                    drawn_samples_in_support = np.vstack([drawn_samples_in_support,redrawn_samples_in_support])

            res[idx,:] = drawn_samples_in_support
        return torch.tensor(res).float()

    def conditionalise(self, condition):
        """Instantiates a new conditional distribution, which can be evaluated
        and sampled from.

        Parameters
        ----------
        condition : 1xD numpy.ndarray or torch.Tensor
            An array of inputs. Inputs set to NaN are not set, and become inputs to
            the resulting distribution. Order is preserved."""
        return ConditionalMDNPosterior(self, condition)

    def sample_from_conditional(self, condition, num_samples):
        """Conditionalises the distribution on the provided condition
        and samples from the the resulting distribution.

        Parameters
        ----------
        condition : 1xD numpy.ndarray or torch.Tensor
            An array of inputs. Inputs set to NaN are not set, and become inputs to
            the resulting distribution. Order is preserved.
        num_samples : int
            The number of samples to draw from the conditional distribution."""
        conditional_posterior = ConditionalMDNPosterior(self, condition)
        samples = cond_posteriori.sample(num_samples)
        return samples

class ConditionalMDNPosterior(MDNPosterior):
    """Wrapperclass for DirectPosterior objects that were trained using
    a Mixture Density Network (MDN) and have been conditionalised.
    Replaces sample, sample_conditional, sample_with_mcmc and log_prob
    functions. Enables the evaluation and sampling of the conditional
    distribution at any arbitrary condition and point.

    Parameters
    ----------
    MDN_Posterior : DirectPosterior
        DirectPosterior object, output of
        inference.build_posterior(density_estimator),
        that was trained with a MDN.
    condition : torch.Tensor
        A vector that holds the conditioned vector. Entries that contain
        NaNs are not set and become inputs to the resulting distribution,
        i.e. condition = [x1, x2, NaN, NaN] -> p(x3,x4|x1,x2).
        """

    def __init__(self, MDN_Posterior, condition):
        self.__class__ = type("ConditionalMDNPosterior",
                              (self.__class__, deepcopy(MDN_Posterior).__class__),
                              {})
        self.__dict__ = deepcopy(MDN_Posterior).__dict__
        self.condition = condition
        self.__conditionalise(condition)


    def __conditionalise(self, condition):
        """Finds the conditional distribution p(X|Y) for a GMM.

        Adpated from code courtesy of @ybernaerts.

        Parameters
        ----------
        condition : 1xD numpy.ndarray or torch.Tensor
            An array of inputs. Inputs set to NaN are not set, and become inputs to
            the resulting distribution. Order is preserved."""

        # revert to the old GMM parameters first
        self.extract_mixture_components()
        self.support = self._prior.support

        pop = self.condition.isnan().reshape(-1)
        condition_without_NaNs = self.condition.reshape(-1)[~pop]

        # check whether the condition is within the prior bounds
        cond_ubound = self.support.upper_bound[~pop]
        cond_lbound = self.support.lower_bound[~pop]
        within_support = torch.logical_and(cond_lbound < condition_without_NaNs,
                                           condition_without_NaNs < cond_ubound)
        if ~torch.all(within_support):
            raise ValueError("The chosen condition is not within the prior support")

        # adjust the dimensionality of the support
        self.support.upper_bound = self.support.upper_bound[pop]
        self.support.lower_bound = self.support.lower_bound[pop]

        not_set_idx = torch.nonzero(torch.isnan(condition))[:,1] #indices for not set parameters
        set_idx = torch.nonzero(~torch.isnan(condition))[:,1] # indices for set parameters
        new_idx = torch.cat((not_set_idx, set_idx)) # indices with not set parameters first and then set parameters
        y = condition[0,set_idx]
        # New centroids and covar matrices
        new_cen = []
        new_ccovs = []
        # Appendix A in C. E. Rasmussen & C. K. I. Williams, Gaussian Processes
        # for Machine Learning, the MIT Press, 2006
        fk = []
        for i in range(self.m.shape[0]):
            # Make a new co-variance matrix with correct ordering
            new_ccov = deepcopy(self.S[i])
            new_ccov = new_ccov[:,new_idx]
            new_ccov = new_ccov[new_idx,:]
            ux = self.m[i,not_set_idx]
            uy = self.m[i,set_idx]
            A = new_ccov[0:len(not_set_idx), 0:len(not_set_idx)]
            B = new_ccov[len(not_set_idx):, len(not_set_idx):]
            #B = B + 1e-10*np.eye(B.shape[0]) # prevents B from becoming singular
            C = new_ccov[0:len(not_set_idx), len(not_set_idx):]
            cen = ux + np.dot(np.dot(C, np.linalg.inv(B)), (y - uy))
            cov = A - np.dot(np.dot(C, np.linalg.inv(B)), C.transpose(1,0))
            new_cen.append(cen)
            new_ccovs.append(cov)
            fk.append(mulnormpdf(y, uy, B)) # Used for normalizing the mc
        # Normalize the mixing coef: p(X|Y) = p(Y,X) / p(Y) using the marginal dist.
        fk = np.array(fk).flatten()
        new_mc = (self.mc*fk)
        new_mc = new_mc / torch.sum(new_mc)

        # set new GMM parameters
        self.m = torch.stack(new_cen)
        self.S = torch.stack(new_ccovs)
        self.mc = new_mc

    def sample_with_mcmc(self):
        """Dummy function to overwrite the existing sample_with_mcmc method."""

        raise DeprecationWarning("MCMC sampling is not yet supported for the conditional MDN.")

    def sample_conditional(self, n_samples, condition=None):
        """Samples from the condtional distribution. If a condition
        is provided, a new conditional distribution will be calculated.
        If no condition is provided, samples will be drawn from the
        exisiting condition.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the conditional distribution.
        condition : 1xD numpy.ndarray or torch.Tensor
            An array of inputs. Inputs set to NaN are not set, and become inputs to
            the resulting distribution. Order is preserved.

        Returns
        -------
        samples : torch.tensor
            Contains samples from the conditional posterior (NxD)
        """

        if condition != None:
            self.__conditionalise(condition)
        samples = self.sample(n_samples)
        return samples

def generate_correlated_parameters(params, corr_mat, pair=(0,1), mag_of_change=0.1):
    """Provided a set of parameters a correlation matrix and pair (i,j) a correlated
    parameter set is generated by changing the pair of parameters along the correlation
    axes, by a specified percentage.


    Parameters
    ----------
    params : torch.Tensor
        A 1D parameter tensor that holds the values of input parameters to the simulator model.
    corr_mat : torch.Tensor
        Matrix containing the pairwise correlation coefficients of the different parameters.
    pair : tuple, list, numpy.ndarray
        Specifies which correlation axis to change the parameters along.
    mag_of_change : float
        Value from the intervall of [-k, k], where k \in R that specifies by how much the
        input parameters are altered.

    Returns
    -------
    new_params : torch.Tensor
        A 1D parameter tensor that differs only for the correlated pair of parameter values
        to the input parameters."""

    i,j = pair
    if i != j:
        corr = corr_mat[i,j]
        new_params = params.detach().clone()
        new_params[:,i] *= (1 + mag_of_change)
        new_params[:,j] *= (1 + mag_of_change * np.sign(corr))
    else:
        # for correlation with itself no change is made
        new_params = params

    return new_params

def sort_stats_by_MSE(stats_sim, stats_obs):
    """Takes two dataframes of summary stats one for the observed
    and one for the simulated data. Sorts simulated summary stats
    according to least MSE vs. the observed summary stats.

    Parameters
    ----------
    stats_sim : pandas.DataFrame
        Contains the summarised results of the simulated traces.
    stats_sim : pandas.DataFrame
        Contains the summarised results of the observed trace."""

    MSE = 1/stats_sim.shape[1]*np.sum(np.square(stats_sim.values - stats_obs.values), axis=1)
    min_MSE = np.sort(MSE) # sort MSE by value
    min_MSE_idx = np.argsort(MSE) # sort indexes of MSEs by value
    #best_params = thetas.iloc[min_MSE_idx]
    return min_MSE_idx, min_MSE

def sort_stats_by_std(stats_sim, stats_obs):
    """Takes two dataframes of summary stats one for the observed
    and one for the simulated data. Sorts simulated summary stats
    according to the distance of sample data points/sample variance
    vs. the observed summary stats.

    Parameters
    ----------
    stats_sim : pandas.DataFrame
        Contains the summarised results of the simulated traces.
    stats_sim : pandas.DataFrame
        Contains the summarised results of the observed trace."""
    sample_var = stats_sim.std()**2
    max_radius = np.max(np.square(stats_obs.values - stats_sim.values)/sample_var.values, axis=1)
    min_rad_idx = np.argsort(max_radius)
    min_rad = np.sort(max_radius)

    return min_rad_idx, min_rad

def best_matches(stats_sim=None, stats_obs=None, simulated_traces=None, trace_obs=None, selected_stats=None, metric="std"):
    """Find the best matching voltage traces compared with the observed trace,
    based on the mean squared error of the summary statistics.

    Parameters
    ----------
    stats_sim = pandas.DataFrame
        Contains the summarised simulation results.
    stats_obs = pandas.DataFrame
        Contains the summarised observation results.
    simulated_traces : list[Trace]
        The simulated data of the voltage traces in the form of Trace() objects.
    trace_obs : Trace
        Trace object for the observed voltage trace
    selected_stats : None, tuple, list or array
            Indices of summary stats to be used.
    metric : str
        Determines which metric to sort the samples by. Currently "MSE"
        and "std" are supported.

    Returns
    -------
    min_MSE_idx : ndarray
        Sorted array containing the indexes of the simulated traces that best
        match the observation based on the MSE. Lowest to highest.
    min_MSE : ndarray
        Array containing the MSE of the simulated traces that best match the
        observation based on the MSE. Lowest to highest.
    min_rad_idx : ndarray
        Sorted array containing the indexes of the simulated traces that best
        match the observation, based on the minimal maximum distance
        based on sample variance. Lowest to highest.
    min_rad : ndarray
        Array containing the MSE of the simulated traces that best match the
        observation, based on the minimal maximum distance
        based on sample variance. Lowest to highest."""


    # calculate summary statitsics
    if type(stats_sim) == type(None) and type(stats_obs) == type(None):
        stats_sim = pd.DataFrame([pd.Series(trace.summarise(selected_stats)) for trace in simulated_traces])
        stats_obs = pd.DataFrame([pd.Series(trace_obs.summarise(selected_stats))])
    if type(stats_obs) == type(None) and type(stats_sim) != type(None):
        stats_obs = pd.DataFrame([pd.Series(trace_obs.summarise(selected_stats))])
    if type(stats_obs) != type(None) and type(stats_sim) == type(None):
        stats_sim = pd.DataFrame([pd.Series(trace.summarise(selected_stats)) for trace in simulated_traces])

    if "m" in metric.lower():
        min_MSE_idx, min_MSE = sort_stats_by_MSE(stats_sim, stats_obs)
        return min_MSE_idx, min_MSE

    if "s" in metric.lower():
        min_rad_idx, min_rad = sort_stats_by_std(stats_sim, stats_obs)
        return min_rad_idx, min_rad

def pairwise_distance(tensor, axis=0, dist="euclidian"):
    """Calculate the pairwise distances for each element in a tensor along a specified axis.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing a list of vectors for which to calculate the pairwise distances.
    axis : int
        Axis along which to calculate the pairwise distances.
    dist : func
        Function that takes two vectors and returns a distance scalar.

    Returns
    -------
    d : numpy.ndarray
        Array containing the pairwise distances, i.e. d[i,j] contains the distance
        between tensor[i] and tensor[j].
        """
    # default distance metrix is euclidian
    if dist == "euclidian":
        dist = lambda x,y: np.linalg.norm((x-y))

    # if axis == 1
    if axis != 0:
        tensor = tensor.T

    d = np.zeros([tensor.shape[0],tensor.shape[0]])

    for i, x in enumerate(tensor):
        for j, y in enumerate(tensor[:i]):
            d[i,j] = dist(x,y)
            d[j,i] = d[i,j]
    return d

def greatest_distance(tensor, axis=0, dist="euclidian"):
    """Calculate the pairwise distances for each element in a tensor along a specified axis
    and return the index of the elements with the greatest distance.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor containing a list of vectors for which to calculate the pairwise distances.
    axis : int
        Axis along which to calculate the pairwise distances.
    dist : func
        Function that takes two vectors and returns a distance scalar.

    Returns
    -------
    np.argmax(D) : numpy.ndarray
        Array containing the indices (i,j) of the greates pairwise distance.
        """
    D = pairwise_distance(tensor, axis, dist)
    return np.unravel_index(np.argmax(D),D.shape)

def gamma(s, a, K, theta_g, theta_s):
    r"""$\gamma(s)$ parameterises a path starting at $\theta_s$ and
    terminating at $\theta_g$ with K sinusoidal basis functions
    with coefficients $a_{n,k}$.

    $\gamma_i(s) = sum_{k=1}^{K}$ a_{i,k} \cdot \sin(\pi k s)
                 + sum_{k=K+1}^{2K}$ a_{i,k} \cdot \sin^2(\pi k s)
                 + (1-s) \cdot theta_s
                 + s \cdot theta_g $

    Parameters
    ----------
    s : float or torch.Tensor
        A value of a tensor of values between 0 and 1. 0 corresponds
        to the starting point in the path, 1 corresponds to the
        endpoint of the path.
    a : torch.Tensor
        Holds $a_{n,k}$, the coefficients of the sinusoidal basis functions
        that we chose to parameterise our path with.
        a.shape = [theta.shape[0], K].
    K : int
        The number of basis functions to parameterise the path with.
    theta_s : torch.Tensor
        Starting point of the path function.
    theta_g : torch.Tensor
        Endpoint of the path function.

    Returns
    -------
    g : torch.Tensor
        Smooth path between theta_s = $\gamma(0)$ and theta_g = $\gamma(1)$
        that is parameterised by sinusoidal basis functions with coefficients
        a."""

    D = theta_g.ndim
    N = theta_g.shape[D-1]
    ks = torch.arange(1,K+1) # number of basis functions

    a.requires_grad_(True)

    # first and second term of gamma
    f1 = a@torch.sin(np.pi*torch.einsum("i,j -> ij", ks, s))
    f2 = a@torch.sin(np.pi*torch.einsum("i,j -> ij", ks+K, s)**2)

    g = f1 + f2 + (1-s)*theta_s.reshape(N,1) + s*theta_g.reshape(N,1)
    return g

def gamma_dot(s, a, K, theta_g, theta_s):
    r"""Derivative of $\gamma(s)$ with respect to s.

    $d\ds gamma_i(s) = sum_{k=1}^{K}$ \pi k a_{i,k} \cdot \cos(\pi k s)
                     + sum_{k=K+1}^{2K}$ 2 \pi k a_{i,k} \cdot \cos(\pi k s) \sin(\pi k s)
                     - theta_s
                     + theta_g$

    Parameters
    ----------
    s : float or torch.Tensor
        A value of a tensor of values between 0 and 1. 0 corresponds
        to the starting point in the path, 1 corresponds to the
        endpoint of the path.
    a : torch.Tensor
        Holds $a_{n,k}$, the coefficients of the sinusoidal basis functions
        that we chose to parameterise our path with.
        a.shape = [theta.shape[0], K].
    K : int
        The number of basis functions to parameterise the path with.
    theta_s : torch.Tensor
        Starting point of the path function.
    theta_g : torch.Tensor
        Endpoint of the path function.

    Returns
    -------
    g : torch.Tensor
        Path derivative between theta_s = $\gamma(0)$ and theta_g = $\gamma(1)$
        that is parameterised by sinusoidal basis functions with coefficients
        a."""

    D = theta_g.ndim
    N = theta_g.shape[D-1]
    ks = torch.arange(1,K+1)

    a.requires_grad_(True)

    # first and second term of d/ds gamma
    f1 = np.pi*ks*a@torch.cos(np.pi*torch.einsum("i,j -> ij", torch.arange(1,K+1), s))
    f2 = 2*np.pi*(ks+K) * a @ (torch.sin(np.pi*torch.einsum("i,j -> ij", ks+K, s))\
                                                *torch.cos(np.pi*torch.einsum("i,j -> ij", ks+K, s)))

    g_dot = f1 + f2 - theta_s.reshape(N,1) + theta_g.reshape(N,1)
    return g_dot

def path_integral(a, K, posterior, x_o, theta_g, theta_s, num_of_points=80):
    r"""Path integral from 0 to 1 which to minimise over the basis coefficients a.

    $L(\gamma) = \int_{0}^{1} -log(p_{\theta|x}(\gamma(s))|x_o)||\dot{gamma}(s)||ds$

    Parameters
    ----------
    a : torch.Tensor
        Holds $a_{n,k}$, the coefficients of the sinusoidal basis functions
        that we chose to parameterise our path with.
        a.shape = [theta.shape[0], K].
    K : sbiDirectPosterior
        The number of basis functions to parameterise the path with.
    posterior: sbi.DirectPosterior
        Posterior object with a method self.log_prob(theta,x_o) that allows for the evaluation
        of the log probability.
    x_o : torch.tensor
        The observation that has been made.
    theta_s : torch.Tensor
        Starting point of the path function.
    theta_g : torch.Tensor
        Endpoint of the path function.
    num_of_points : int
        Specifies the number of points on the path over which to integrate.

    Returns
    -------
    L : torch.Tensor
        The value of the path integral, evaluated over the specified number of
        points between 0 and 1."""

    points = torch.linspace(0,1,num_of_points)
    ds = points[1]-points[0]

    a.requires_grad_(True)

    g = gamma(points, a, K, theta_g=theta_g, theta_s=theta_s)
    g_dot = gamma_dot(points, a, K, theta_g=theta_g, theta_s=theta_s)

    posterior.default_x = x_o

    log_p = posterior.log_prob(g.T)
    log_p[log_p == float("-inf")] = -1e10 # replace -inf with large value for the loss to be tractable

    L = torch.sum(-log_p*torch.norm(g_dot)*ds)
    return L

def fit_high_posterior_path(theta_s, theta_g, posterior, x_o, init_size=1e-2,
                            n_components=2, n_points=80, n_itters=1500,
                            l_rate=0.001, plot_loss=True, loss_scale="log"):
    r"""Fit a path along regions of high posterior, between two points
    [theta_s, theta_g] by minimizing a path integral L(\gamma) over the
    intervall [0,1] using gradient descent.

    $\gamma_i(s) = sum_{k=1}^{K}$ a_{i,k} \cdot \sin(\pi k s)
                 + sum_{k=K+1}^{2K}$ a_{i,k} \cdot \sin^2(\pi k s)
                 + (1-s) \cdot theta_s
                 + s \cdot theta_g $

    $L(s) = \int_{0}^{1} -log(p_{\theta|x}(\gamma(s))|x_o)||\dot{gamma}(s)||ds$

    Parameters
    ----------
    theta_s : torch.Tensor
        Starting point of the path function.
    theta_g : torch.Tensor
        Endpoint of the path function.
    posterior: sbi.DirectPosterior
        Posterior object with a method that allows for the evaluation
        of the log probability.
    x_o : torch.tensor
        The observation that has been made.
    init_size : float
        Determines the magnitude of the random numbers during initialisation
        of a. Numbers too large might lead to posterior.log_prob = -inf.
        Hinders the fitting process.
    n_components : int
        Number of basis functions and coefficients to use in the parameterisation
        of the path $\gamma(s)$.
    n_of_points : int
        Specifies the number of points on the path over which to integrate.
    n_itters : int
        Number of itterations of the gradient descent.
    l_rate = float
        Stepsize of for gradient updates.
    plot_loss : bool or str
        Whether to plot the evolution of the loss during the fitting process,
        after the fitting is done.
        If "live" is set and the cell is run with %matplotlib notebook,
        the plot is updated in real time, however this way it also runs
        much slower.
    loss_scale : str
        Whether to plot the loss on a "linear" or on a "log" scale.
        Argument gets passed to plt.yscale().

    Returns
    -------
    path : torch.Tensor
        The fitted optimal high posterior path $\gamma(s)$ with the
        fitted coefficients $a_{n,k}$"""

    N = theta_s.shape[0]
    a = init_size*torch.rand((N,n_components))

    # optimizer
    optimizer = torch.optim.Adam([a], lr=l_rate)

    itters = np.arange(1,n_itters+1)
    losses = []

    if plot_loss == "live":
        # for real time plotting functionality
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("Gradient Descent")
        plt.ylabel("loss")
        plt.xlabel("itteration")
        plt.ion()

        fig.show()
        fig.canvas.draw()

    # gradient descent
    for i in itters:
        print("\rProgress: ({}/{})".format(i,n_itters), end="")
        optimizer.zero_grad()
        loss = path_integral(a,n_components, posterior, x_o, theta_g, theta_s)
        losses.append(loss)
        loss.backward()
        optimizer.step()

        if plot_loss == "live":
            ax.clear()
            #ax.set_xlim(0,n_itters)
            ax.plot(np.arange(i),losses, ".-")
            fig.canvas.draw()

    if plot_loss:
        plt.figure(figsize=(10,5))
        plt.title("Gradient Descent")
        plt.plot(itters,losses, ".-")
        plt.yscale(loss_scale)
        plt.grid(True)
        plt.ylabel("loss")
        plt.xlabel("itteration")

    path = gamma(torch.linspace(0,1,n_points),a,n_components, theta_g=theta_g, theta_s=theta_s).detach().T
    return path

def mulnormpdf(X, mu, cov):
    """Evaluates the PDF for the multivariate Guassian distribution.

    Parameters
    ----------
    X : np array
        Inputs/entries row-wise. Can also be a 1-d array if only a
        single point is evaluated.
    mu : nparray
        Center/mean, 1d array.
    cov : 2d np array
        Covariance matrix.

    Returns
    -------
    prob : 1d np array
        Probabilities for entries in `X`.
    """

    # Evaluate pdf at points or point:
    if X.ndim == 1:
        X = torch.atleast_2d(X)
    sigma = torch.atleast_2d(cov) # So we also can use it for 1-d distributions

    N = mu.shape[0]
    ex1 = torch.inverse(sigma)@(X-mu).T
    ex = -0.5 * (X-mu).T * ex1
    if ex.ndim == 2: ex = torch.sum(ex, axis = 0)
    K = 1 / torch.sqrt ( torch.pow(2*torch.tensor(np.pi), N) * torch.det(sigma) )
    return K*torch.exp(ex)

def compare_correlated_summary_stats(params, syn_current, pair, corr_mat, mag_of_change, selected_indices, using="C++", mode="auto", summary_func=None):
    """Simulates a reference and a correlated trace. Summarises both and returns an DataFrame
    containing the selected summary statistics.

    Parameters
    ----------
    params : torch.Tensor
        A 1D parameter tensor that holds the values of input parameters to the simulator model.
        Contains the initial current "V0" [float], the stimulation current trace
        "It" [ndarray], the product of the stimulation current trace and the area
        of the soma "ItA": ndarray, the time axis "ts": ndarray, the step size
        "dt": float and the simulus time course "StimOnset": float,
        "StimEnd": float and "StimDur": float.
    pair : tuple, list, numpy.ndarray
        Specifies which correlation axis to change the parameters along.
    corr_mat : torch.Tensor
        Matrix containing the pairwise correlation coefficients of the different parameters.
    mag_of_change : float
        Value from the intervall of [-k, k], where k \in R that specifies by how much the
        input parameters are altered.
    selected_stats : None, tuple, list or array
        Indices of summary stats to be used.
    sumary_func : function or str
        A function that takes voltage trace V(t), current trace I(t), time
        points t and time step dt and outputs a dictionary with summary stats
        and descriptions.
        Alternatively a descriptive string like "v1" or "v2" can be provided to
        use one of the summary methods already implemented as part of Trace().
    mode : str
        Specifies whether cell/experiment specific parameters like ASoma and V0
        should be specified as part of the model parameters ("model_params")
        or if ASoma and V0 should be considered part of the syn_current
        or trace object ("trace").

    Returns
    -------
    sum_df : pandas.DataFrame
        Contains the summary statistics for both the base and correlated trace."""

    # generate correlated parameters
    pars = generate_correlated_parameters(params, corr_mat, pair=pair, mag_of_change=mag_of_change)

    # run simulation with reference and correlated params
    corr_trace = runHH(pars, syn_current=syn_current, using=using, mode=mode)
    base_trace = runHH(params, syn_current=syn_current, using=using, mode=mode)

    # collect results of summaries in DataFrame
    sum_corr_trace_df = pd.Series(
        corr_trace.summarise(selected_indices, summary_func=summary_func), name="correlated params")
    sum_base_trace_df = pd.Series(base_trace.summarise(selected_indices,summary_func=summary_func), name="base params")

    sum_df = pd.concat([sum_base_trace_df,sum_corr_trace_df], axis=1)

    return sum_df
