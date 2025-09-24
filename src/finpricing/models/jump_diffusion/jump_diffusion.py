"""
Jump-Diffusion Models: Merton and Kou
"""
import numpy as np
from scipy.integrate import quad_vec
from finpricing.parameters import VIXMertonModelParameters, VIXKouModelParameters


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               DRIFT TERMS                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def shared_drift_base(T: float, omega: float, k1: float) -> float:
    """
    Computes the shared drift base term for jump-diffusion models.
    """
    return -0.5 * (omega**2) * (1 - np.exp(-2 * k1 * T)) / (2 * k1)


def merton_integrand(s: float, T: float, k2: float, m: float, delta: float) -> float:
    """
    Drift integrand for the Merton model.
    """
    return np.exp(np.exp(-k2 * (T - s)) * m + 0.5 * np.exp(-2 * k2 * (T - s)) * delta**2) - 1


def kou_integrand(
        s: float,
        T: float,
        k2: float,
        p: float,
        alpha_plus: float,
        alpha_minus: float
) -> float:
    """
    Drift integrand for the Kou model.

    Args:
        s (float): Time s.
        T (float): Time T.
        k2 (float): Jump intensity.
        p (float): Jump probability.
        alpha_plus (float): Positive jump
        alpha_minus (float): Negative jump

    Returns:
        float: Value of the integrand at time s.
    """
    return (p * alpha_plus / (alpha_plus - np.exp(-k2 * (T - s))) 
            + (1 - p) * alpha_minus / (alpha_minus + np.exp(-k2 * (T - s))) - 1)


def merton_drift(T: float, omega: float, lmbd: float, k1: float, k2: float, m: float, delta: float) -> float:
    """
    Computes the drift term for the Merton jump-diffusion model.
    """
    base_drift = shared_drift_base(T, omega, k1)
    integrand = lambda s: merton_integrand(s, T, k2, m, delta)
    jump_term = lmbd * quad_vec(integrand, 0, T)[0]
    return base_drift - jump_term


def kou_drift(T: float, omega: float, lmbd: float, k1: float, k2: float, p: float, alpha_plus: float, alpha_minus: float) -> float:
    """
    Computes the drift term for the Kou jump-diffusion model.
    """
    base_drift = shared_drift_base(T, omega, k1)
    integrand = lambda s: kou_integrand(s, T, k2, p, alpha_plus, alpha_minus)
    jump_term = lmbd * quad_vec(integrand, 0, T)[0]
    return base_drift - jump_term





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               JUMP TERMS                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def simul_total_jumps(
        T1: float,
        T2: float,
        num_paths: int,
        model_params):
    """
    Simulate jump terms according to model parameters between times T1 and T2.

    Args:
        T1 (float): Start time (non-negative).
        T2 (float): End time (positive).
        num_paths (int): Number of simulation paths.
        model_params (VIXMertonModelParameters or VIXKouModelParameters): Model parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 1D arrays representing:
            - Total jumps for each path: \sum Y_j
            - Exponential-weighted jumps for each path: \sum e^{k2 tau_j} Y_j 
    """
    if T1 < 0 or T2 <= T1:
        raise ValueError("T1 must be non-negative and T2 must be greater than T1.")

    # Validate and extract parameters
    if isinstance(model_params, VIXMertonModelParameters):
        lmbd = model_params.lmbd
        k2 = model_params.k2
        m = model_params.m
        delta = model_params.delta
        model_type = "Merton"
    elif isinstance(model_params, VIXKouModelParameters):
        lmbd = model_params.lmbd
        k2 = model_params.k2
        p = model_params.p
        alpha_plus = model_params.alpha_plus
        alpha_minus = model_params.alpha_minus
        model_type = "Kou"
    else:
        raise ValueError("Unsupported model parameters. Must be VIXMertonModelParameters or VIXKouModelParameters.")

    # Simulate jump counts and times
    jump_counts = np.random.poisson(lmbd * (T2 - T1), size=num_paths)
    max_jumps = np.max(jump_counts)
    jump_times = np.random.uniform(low=T1, high=T2, size=(num_paths, max_jumps))

    # Generate jump sizes based on the model
    if model_type == "Merton":
        jump_sizes = np.random.normal(loc=m, scale=delta, size=(num_paths, max_jumps))

    elif model_type == "Kou":
        jump_directions = np.random.choice([1, -1], size=(num_paths, max_jumps), p=[p, 1 - p])
        jump_sizes = jump_directions * np.random.exponential(
            scale=(jump_directions > 0) * 1 / alpha_plus + (jump_directions < 0) * 1 / alpha_minus,
            size=(num_paths, max_jumps),
        )

    # Compute total jumps and exponential-weighted jumps
    total_jumps = np.sum(jump_sizes * (jump_counts[:, None] > 0), axis=1)
    exp_weighted_jumps = np.sum(jump_sizes * np.exp(k2 * jump_times) * (jump_counts[:, None] > 0), axis=1)

    return total_jumps, exp_weighted_jumps



# A function to simulates total jumps for each tenor period at once

# both for the index and the variance swaps/VIX


def simul_tenor_jumps(model_params, num_paths = 10**5, tenor_dates = np.arange(0, 1/2, 1/12)):
    """
    Simulates the sum of jumps for each tenor period for the index 
    and 
    the total jump term the varaince for swaps up to each tenor date.

    Arguments:
        tenor_dates - sorted numpy list of tenor dates
        
        model_params - an object of class vix_model_parameters 
                       containing the parameters of the jumps process and the type

    Returns:
        - \sum_{T_{i-1}=<tau_j<T_i} Y_j

        and 

        _ \sum_{tau_j<T_i} exp(-k2*(T_i -tau_j)) Y_j
    """
    
    #we need the k2 parameter
    k2 = model_params.k2
    
    
    #2D arrays of jumps in each tenor peirod for each simulation
    total_jumps = np.zeros((num_paths, len(tenor_dates)-1))
    vix_jumps = np.zeros((num_paths, len(tenor_dates)-1))
    
    for i in range(len(tenor_dates)-1):
        T1 = tenor_dates[i]
        T2 = tenor_dates[i+1]
        total_jumps[:, i], vix_jumps[:, i] = simul_total_jumps(T1, T2, num_paths, model_params)

    return total_jumps, np.exp(-k2*tenor_dates[1:])*np.cumsum(vix_jumps, axis=1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               DIFFUSION TERMS                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def simulate_tenor_diffusions(num_paths: int, tenor_dates: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the diffusion parts of VIX and the index on tenor dates.
    
    Args:
        num_paths: The number of Monte Carlo simulation paths
        tenor_dates: Sorted numpy array of tenor dates
        k1: Volatility scaling parameter
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays of shape (num_paths, len(tenor_dates) - 1):
            - ZV: Diffusion term in volatility swap dynamics from Cont&Kokholm equation (3.1)
                  ZV_i = \\int_{0}^{T_i}exp(k1*(T_i-s))dZ_s
            - dZ: Increments of the diffusion controlling the index prices
                  Each column represents Z(T_i) - Z(T_{i-1})
    """
    # Compute variances of increments \int_{T_{i-1}}^T_i exp(k1*s) dZ_s, i = 1, ..., m
    variations = np.diff(np.exp(2*k1*tenor_dates))/(2*k1)

    # Compute correlation of Z(T_i) - Z(T_{i-1}) and \int_{T_{i-1}}^T_i exp(k*s) dZ_s
    correlations = np.diff(np.exp(k1*tenor_dates))/k1

    # Simulate independent normal variables for \int_{T_{i-1}}^T_i exp(k*s) dZ_s
    dIZ = np.random.randn(num_paths, len(tenor_dates)-1)*np.sqrt(variations)

    # Simulate W(T_i) - W(T_{i-1})
    dZ = np.random.randn(num_paths, len(tenor_dates)-1)
    
    dZ = (correlations/variations)*dIZ + \
         np.sqrt(np.diff(tenor_dates) - correlations**2/variations)*dZ

    ZV = np.exp(-k1*tenor_dates[1:])*np.cumsum(dIZ, axis=1)

    return ZV, dZ
