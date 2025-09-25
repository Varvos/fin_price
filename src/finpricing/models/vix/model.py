"""
VIX Model implementation for simultaneous pricing of S&P and VIX derivatives
"""
import numpy as np
from finpricing.parameters import VIXMertonModelParameters, VIXKouModelParameters
from finpricing.models.jump_diffusion.jump_diffusion import simul_tenor_jumps, simulate_tenor_diffusions


def get_vix_prices(Vi_0: np.ndarray, diff_V: np.ndarray, jump_V: np.ndarray, 
                  tenor_dates: np.ndarray, model_params) -> np.ndarray:
    """
    Given the simulations of diffusion terms and jump terms of the variance swaps
    and the initial values V^0_{T_0}, V^1_{0}, ..., V^{m-1}_{0}
    returns the values at the corresp. tenor dates V^1_T_1, ..., V^{m-1}_T_{m-1}

    Args:
        Vi_0: Initial values of V^i_0 i= 1, ..., m-1
        diff_V: Realised diffusion terms up to each tenor date
        jump_V: Realised jump terms up to each tenor date  
        tenor_dates: Sorted array of tenor dates
        model_params: Model parameters object
        
    Returns:
        VT_i: Values at tenor dates
    """
    from finpricing.utils.vix_utils import vix_drift_coeff
    
    # Get the constant drift terms \int_0^T_i mu_i(t)dt
    m = len(tenor_dates)-1
    tenor_drifts = np.zeros(m-1)
    
    for i in range(m-1):
        tenor_drifts[i] = vix_drift_coeff(tenor_dates[i+1], model_params)
    
    VT_i = Vi_0[0]*np.ones((len(diff_V), len(Vi_0)))
    
    VT_i[:,1:] = Vi_0[1:]*np.exp(tenor_drifts + model_params.omega*diff_V + jump_V)
                   
    return VT_i


def get_sigmas(Vi0, VTi, constraint_cf, b_i):
    """
    Compute the tenor volatilities sigma_i, from the initial variance swap prices
    and given realisations/simulations of the variance swap prices at tenor dates.

    Args:
        Vi0: Initial values of V^i_0
        VTi: Realisations/simulations of the variance swap prices at tenor dates
        constraint_cf: Coefficient in the formula (σ_i)^2 = V^i_{T_i} - lambda* b_i^2 * (constraint_cf)
        b_i: Model coefficients for the index jumps
        
    Returns:
        sigmas_sq: Squared volatility coefficients of the index diffusion
    """
    # Initialize with zeros
    sigmas_sq = VTi - (constraint_cf) * (VTi/Vi0) * (b_i**2)
    return sigmas_sq


def get_UTm(Vi_0, VT_i, diff_Z, jumps_S, sigmas_sq, b_i, tenor_dates, model_params):
    """
    Given the simulations/realisations of diffusion terms and jump terms of the index,
    the variance swap prices, the sigma_i squared and the b_i coefficients returns the values
    of the auxiliary price U_{T_k}.

    Args:
        Vi_0: Initial values of V^i
        VT_i: Realisations/simulations of the variance swap prices at tenor dates
        diff_Z: Realisations/simulations of the increments of the diffusion controlling the index prices
        jumps_S: Realised total jump of the index in each tenor period
        sigmas_sq: Squared volatility coefficients of the index diffusion
        b_i: Model coefficients for the index jumps
        tenor_dates: Array of tenor dates
        model_params: Model parameters object
        
    Returns:
        U_{T_k}: Auxiliary price (see page 259 from Cont&Kokhlom)
    """
    # Compute the coefficients in u_i(., x)
    ui_factors = np.sqrt(VT_i/Vi_0)*b_i

    model_type = model_params.model_type
    # Compute the integral ∫(e^(u_i(., x)) - 1) v(dx) in the drift term
    lmbd = model_params.lmbd
    
    # Unpack the model parameters
    if model_type == 'Merton':
        m, delta = model_params.m, model_params.delta
        integral = lmbd*(np.exp((m*ui_factors + 0.5*(delta**2)*(ui_factors**2))) - 1)
        
    elif model_type == 'Kou':
        p, a_plus, a_minus = model_params.p, model_params.alpha_plus, model_params.alpha_minus
        integral = lmbd*(p*a_plus/(a_plus-ui_factors) + (1-p)*a_minus/(a_minus+ui_factors) - 1)

    drift_term = np.sum(-(0.5*(model_params.rho**2)*sigmas_sq + integral)*np.diff(tenor_dates), axis=1)

    diffusion_term = model_params.rho*np.sum(diff_Z * np.sqrt(sigmas_sq), axis=1)

    jump_term = np.sum(jumps_S * ui_factors, axis=1)

    return np.exp(drift_term+diffusion_term+jump_term)


class VixModel:
    """
    A class for a model for simultaneous pricing of S&P and VIX derivatives.
    
    Args:
        tenor_dates: Array of tenor dates (in years) starting with 0, 
                     defaulted to six month range np.arange(0, 1/2, 1/12)
        model_type: String; the type of the model used for the jumps, either 'Merton' or 'Kou'
    """
    def __init__(self, tenor_dates=np.array([0, 1, 2, 3, 4, 7, 10])/12, model_type='Merton'):
        self.model_type = model_type
        self.tenor_dates = tenor_dates
        
        # The initial variance swap rates V^0_0, ..., V^0_{m-1}
        # Initialized to the values from the paper
        self.Vi_0 = np.array([0.041, 0.052, 0.056, 0.059, 0.062, 0.061])
        
        # Monte Carlo data for tenors used to price Index Options
        self._stored_data = []
        
        # Model parameters
        if model_type == 'Merton':
            self.params = VIXMertonModelParameters()
        elif model_type == 'Kou':
            self.params = VIXKouModelParameters()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if model_type == 'Merton':  # Gaussian jumps model
            self.b_i = np.array([-0.140, -0.161, -0.162, -0.187, -0.198, -0.199])
            self.constraint_cf = self.params.lmbd*(self.params.m**2+self.params.delta**2)
            
        elif model_type == 'Kou':  # Double exponential jumps
            self.b_i = np.array([-0.141, -0.159, -0.158, -0.187, -0.195 , -0.192])
            self.constraint_cf = 2*self.params.lmbd*(self.params.p/(self.params.alpha_plus**2)+
                                    (1-self.params.p)/(self.params.alpha_minus**2))
        else:  # Other model
            raise ValueError("Unknown model")
        
    
    def store_tenor_data(self, num_paths=10**5):
        """
        Simulate and store jumps and diffusion terms per each tenor period.
        
        Args:
            num_paths: Number of Monte Carlo paths to simulate
        """
        # Diffusions
        diff_V, diff_Z = simulate_tenor_diffusions(num_paths, self.tenor_dates, k1=self.params.k1)

        # Jumps
        jumps_S, jumps_V = simul_tenor_jumps(self.params, num_paths, self.tenor_dates)

        # Store the corresponding V^i_T_i values for i = 0, ..., m-1
        VTi = get_vix_prices(self.Vi_0, diff_V[:,:-1], jumps_V[:,:-1], self.tenor_dates, self.params)
        
        self._stored_data = [diff_V, diff_Z, jumps_S, jumps_V, VTi]
        
    
    def get_index_prices(self):
        """
        Get index prices (placeholder method).
        """
        pass

    def vix_option_pricer(self, S0, K, T, r, option_type='call', pricing_method='Kar&Madan'):
        """
        Price VIX options using Fourier transform methods.
        
        Args:
            S0: Current VIX level
            K: Strike price  
            T: Time to maturity
            r: Risk-free rate
            option_type: Option type ('call' or 'put')
            pricing_method: Fourier pricing method ('Cont&Tankov' or 'Kar&Madan')
            
        Returns:
            Option price
        """
        from finpricing.pricing_methods.vix_pricing import price_vix_options
        return price_vix_options(S0, K, T, r, self.params, pricing_method, option_type)
    
    def index_option_pricer(self, S0, strikes, tenor_index, r=0., option_type='call'):
        """
        Price index options using Monte Carlo simulation.
        
        Args:
            S0: Spot price
            strikes: Array of strike prices
            tenor_index: Index of the expiry date in tenor dates
            r: Interest rate
            option_type: Type of the option either 'call' or 'put'
            
        Returns:
            Array of option prices
        """
        if self._stored_data == []:
            self.store_tenor_data()
        
        from finpricing.pricing_methods.vix_pricing import price_index_options
        return price_index_options(S0, self.Vi_0, strikes, r, self.tenor_dates[:tenor_index+1],
                                  self._stored_data, self.b_i, self.params, option_type)

    
    def fit_data(self, data):
        """
        Fit model parameters to market data (placeholder method).
        
        Args:
            data: Market data to fit
        """
        pass