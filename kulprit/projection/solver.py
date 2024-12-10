"""optimization module."""
import arviz as az
import numpy as np
from scipy.optimize import minimize


def solve(model_log_likelihood, pps, ref_idata, initial_guess, var_info):
    """The primary projection method in the procedure.

    Parameters:
    ----------
    model_log_likelihood: Callable
        The log-likelihood function of the model
    pps: array
        The predictions of the reference model
    initial_guess: array
        The initial guess for the optimization
    var_info: dict
        The dictionary containing information about the size and transformation of the variables

    Returns:
    -------
        new_idata: arviz.InferenceData
        loss: float
    """
    num_samples = len(pps)
    posterior_array = np.zeros((num_samples, len(initial_guess)))
    posterior_dict = {}
    objectives = []
    #posterior = az.extract(ref_idata, num_samples=num_samples)


    # size = 0
    # index = []
    # predefined0 = []
    # predefined1 = []
    # for key, values in var_info.items():
    #     shape, new_size, transformation = values
    #     if ("|" in key and "_sigma" in key):
    #         index.append(size + new_size-1)
    #         if "1" in key:
    #             predefined0.append(0.71318386)
    #             predefined1.append(1.16124516)

    #         if "Time" in key:
    #             predefined0.append(-0.6520932)
    #             predefined1.append(-0.28910714)
    #     size += new_size


    # def prob_bound0(params, index, predefined0):
    #     #print(params[index])
    #     loss = (params[index]- predefined0) 
    #     return loss
    
    # def prob_bound1(params, index, predefined1):
    #     #print(params[index])
    #     loss = (predefined1 - params[index]) 
    #     return loss

    # if index:
    #     index = np.array(index)
    #     predefined0 = np.array(predefined0)
    #     constraints = [ {
    #         "type": "ineq",
    #         "fun": prob_bound0,
    #         "args": (index, predefined0),
    #     },
    #      {
    #         "type": "ineq",
    #         "fun": prob_bound1,
    #         "args": (index, predefined1),
    #     }
    #     ]
    # if index:
    #     constraints =  {'type': 'ineq', 'fun': lambda x: -np.sum(x[index])}
    # else:
    #     constraints = None


    for idx, obs in enumerate(pps):
        opt = minimize(
            model_log_likelihood,
            args=(obs),
            x0=initial_guess,
            tol=0.001,
            #constraints=constraints,
            method="SLSQP",
        )

        posterior_array[idx] = opt.x
        objectives.append(opt.fun)
        initial_guess = opt.x

    size = 0
    for key, values in var_info.items():
        shape, new_size, transformation = values
        posterior_dict[key] = posterior_array[:, size : size + new_size].reshape(
            1, num_samples, *shape
        )
        if transformation is not None:
            posterior_dict[key] = transformation(posterior_dict[key])
        size += new_size

    # for key, values in var_info.items():
    #     if "|" in key and "_sigma" in key:
    #         posterior_dict[key] = np.std(posterior_dict[key.split("_sigma")[0]], axis=1)


    new_idata = az.from_dict(posterior=posterior_dict)
    loss = np.mean(objectives)
    return new_idata, loss
