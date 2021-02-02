import numpy as np
import torch
import matplotlib.pyplot as plt

# sbi
import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils import pairplot, conditional_pairplot, conditional_corrcoeff, eval_conditional_density

# setting random seed
np.random.seed(1)

def simulator_placeholder(params):
    """Noisy Identity.
    Takes a vector of 2 parameters and adds noise."""
    
    return np.random.multivariate_normal(np.array(params),np.eye(2))
    
# mock observation
x_o = np.array([ 10., 50.])

# pre-simulating data
prior_min = [0.1, 0.001]
prior_max = [100., 100.]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

prior_sample = prior.sample((2000,))

thetas = []
outputs = []
for params in prior_sample:
    result = simulator_placeholder(params)
    outputs.append(result)
    thetas.append(params)

thetas = torch.stack(thetas, dim=0)
outputs = np.array(outputs)

# setting up sbi
simulator, prior = prepare_for_sbi(simulator_placeholder, prior)

density_estimator_build_fun = posterior_nn(model='mdn')

inference = SNPE(simulator, prior, density_estimator=density_estimator_build_fun, 
                 show_progress_bars=True, num_workers=2)
                 
inference.provide_presimulated(torch.as_tensor(thetas, dtype=torch.float32), \
                               torch.as_tensor(outputs, dtype=torch.float32), from_round=0)

# running the inference                               
proposal = None
posterior = inference(num_simulations=0, proposal=proposal)

# sampling a condition
posterior.set_default_x(x_o)
condition = posterior.sample((1,))

# compute conditional correlations
cond_coeff_mat = conditional_corrcoeff(
    density=posterior,
    condition=condition,
    limits=torch.tensor(list(zip(prior_min,prior_max))))
fig, ax = plt.subplots(1,1, figsize=(4,4))
im = plt.imshow(cond_coeff_mat, cmap='PiYG') # without clim=[-1,1]
_ = fig.colorbar(im)
plt.show()

print(cond_coeff_mat)

