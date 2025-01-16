"""
A module for pricing options using Fourier methods.
"""

import numpy as np

def fourier_call_pricer(chr_func, T, r, s_0=0, method='Cont&Tankov'):
    """
    Returns the prices of call options for a list of log moneyness valuues centered around exp(s_0),
    for an asset with given characteristic function, using a Fourier transform methods.
    
    chr_func - characteristic function of the asset
    T - positive float, the expiry (in years)
    r - positive float, the interest rate
    s_0 - a float, the log of the moneyness 
    method - the method used for for fuourier tansport, either Cont&Kokhlom or Kar&Madan
    """
    
    #initialising hyperparameters for the Fourier Transform (FT) method
    N = 1024
    Delta = 0.17
    sgm = 0.3575
    bs_model_params = vix_model_parameters('Black-Scholes')
    bs_model_params.sigma = sgm
    
    d = 2*np.pi/(Delta*N)
    A = Delta*(N-1)

    d = 2*np.pi/(Delta*N)
    A = Delta*(N-1)

    #integration grid x_k = -A/2 + k*Delta, k=0, ..., N-1
    xs = np.array([-0.5*A + k*Delta for k in range(N)])

    #Quadrature weights:
    w = np.ones(N)
    w[0] = w[-1] = 0.5
    
    #Simpson's rule quadrature weights
#     w = np.array([3-(-1)**j for j in range(N)])
#     w[0] -= 1
#     w[-1] -=1
#     w = w/3   

    #log moneyness grid u_n = -N*d/2+s_0 + n*d, n = 0, ..., N-1
    us = np.array([-N*d/2+s_0 + n*d for n in range(N)])
    
    if method == 'Cont&Tankov':
        #compute the FT of modified option price
        zeta_T = np.exp(-r*T)*chr_func(xs-1j)
        zeta_T -= np.exp(1j*r*T*xs)*char_func(xs-1j, T, bs_model_params)
        zeta_T /=(1j*xs - xs*xs)

        #compute the inverse fourier transfroorm using FFT
        zs = fft(zeta_T*w*np.exp(-1j*(s_0-N*d/2)*(xs+0.5*A)))
        zs = Delta*np.exp(1j*A*us/2)*zs/(2*np.pi)
        
        #recover the option prices
        ops = zs.real + bs_call(1, np.exp(us), T, r, sgm)
        
    elif method == 'Kar&Madan':
        #an exponent s.t. ES^{alf+1}<+inf
        alf = 0.75
        
        xs = np.array([Delta*k for k in range(N)])
        
        #Fourier Transform of the modified option price
        zeta_T = np.exp(-r*T)*chr_func(xs-(alf+1)*1j)
        zeta_T /= (alf**2+alf-xs*xs+1j*(2*alf+1)*xs)
        
        #Simpson's rule quadrature weights
        w = np.array([3-(-1)**j for j in range(N)])
        w[0] -= 1
        w = w/3
        
        #compute the inverse fourier transfroorm using FFT
        zs = fft(zeta_T*np.exp((N*d/2-s_0)*1j*xs)*w)*Delta
        
        #recover the option prices
        ops = np.exp(-alf*us)*zs.real/(np.pi)
        
    else:#other method
        raise ValueError("Unkown method")
    #return 
    return us, ops
    