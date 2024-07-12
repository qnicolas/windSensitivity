import numpy as np
import xarray as xr
from scipy.signal import hilbert
from scipy.integrate import cumtrapz

cp = 1004. # Heat capacity of air in J/kg/K
Lc = 2.5e6 # latent heat of condensation in J/kg
g = 9.81   # m/s2
Rv = 461   # Gas constant for water vapor, J/kg/K 
Rd = 287   # Gas constant for dry air, J/kg/K 

def topographic_profile(kind,a=100e3,A=1000):
    """Computes one of three default topographic profiles (Witch of Agnesi, Gaussian ridge & truncated cosine ridge)
    args:
     - kind, type of mountain shape (either agnesi, gaussian or cos)
     - a, mountain half-width in m
     - A, mountain height in m
    returns:
     - xx, x-grid in m
     - hx, topographic profile in m
     """
    xx=np.arange(-10000e3,20000e3,5e3)
    if kind=='gaussian':
        hx = A*np.exp(-xx**2/2/(a/2)**2)
    elif kind=='cos':
        hx = A/2*(1+np.cos(np.pi*np.maximum(-1,np.minimum(1,xx/a))))
    elif kind=='agnesi':
        hx = A*a**2/(xx**2+a**2)
    return xx,hx

#######################################################################################
############################  LINEAR MOUNTAIN WAVE THEORY  ############################
#######################################################################################
def k_vector(Nx,dx):
    """Given an x grid, return the grid of wavenumbers on which an FFT will be computed
    args:
     - Nx, number of points in grid
     - dx, grid spacing
    returns:
     - k, wavenumber array
    """
    return 2*np.pi*np.fft.fftfreq(Nx,dx)

def m_exponent_damped(k,N,U,xi,hydrostatic=False):
    """Vectorized function to compute the vertical wavenumbers in linear mountain wave theory w/ damping, constant N & U
    args:
     - k : array-like, horizontal wavenumber [m^-1]
     - N : float, Brunt-Vaisala frequency [s^-1]
     - U : float, basic-state wind [m s^-1]
     - xi : float, damping rate [s^-1]
     - hydrostatic : bool, self-descriptive
    returns:
     - m : array-like, vertical wavenumber
    """
    # Scorer parameter
    lz2 = N**2/U**2/(1-1j*xi/(k*U))
    
    sgnk = np.sign(k)
    if hydrostatic:
        k=1e-10
    # m in the evanescent case
    m1 = (np.real(lz2) < k**2 ) * 1j*np.sqrt(k**2-lz2 + 0.j) 
    # m in the vertically propagating case
    m2 = (np.real(lz2) >= k**2) * sgnk *np.sqrt(lz2-k**2 + 0.j)
    return m1 + m2

def linear_eta_damped(x,z,h,N,U,xi=1./86400.,hydrostatic=False):
    """Computes vertical displacement in linear mountain wave theory w/ damping, constant N & U
    args:
     - x : 1D np.array, horizontal coordinate [m]
     - z : 1D np.array, vertical coordinate [m]
     - h : 1D np.array, surface height [m]
     - N : float, Brunt-Vaisala frequency [s^-1]
     - U : float, basic-state wind [m s^-1]
     - xi : float, damping rate [s^-1]
     - hydrostatic : bool, self-descriptive
    returns:
     - eta : 2D xr.DataArray, vertical displacement in [m]
    """
    # Horizontal wavenumber
    k=k_vector(len(x),x[1]-x[0]); k[0] = 1e-10 # dummy nonzero value
    
    # Fourier-transformed terrain
    h_hat = np.fft.fft(h); h_hat[0] = 0.
    
    # Fourier-transformed vertical displacement
    etahat = h_hat[:,None]*np.exp(1j * m_exponent_damped(k[:,None],N,U,xi,hydrostatic=hydrostatic)  *  z[None,:])
    
    # Temperature profile & pressure profile
    Tz = 300. + (300.*N**2/9.81 - 9.81/1004.)*z
    lnpz = - cumtrapz(9.81/(287.*Tz),z,initial=0.)
    
    eta=xr.DataArray(np.real(np.fft.ifft(etahat,axis=0)),coords={'x':x/1e3,'z':z,'p':('z',1000.*np.exp(lnpz))},dims={'x':x/1e3,'z':z})
    return eta

########################################################################################
############################  NICOLAS & BOOS (2022) THEORY  ############################
########################################################################################
def compute_Lq(Ms_ov_M,U,tauq):
    tauqtilde = 0.6*tauq # conversion from full-tropospheric average to lower tropospheric average, see Ahmed et al 2020
    return Ms_ov_M*U*tauqtilde

def linear_precip_theory_damped(x,h,N,U,xi=1./86400.,tauT=7.5 * 3600,tauq=27.5 * 3600,switch=1,zbot=1000,ztop=4000,hydrostatic=False):
    """Computes the orographic precipitation perturbation predicted by the linear theory (equation (3) in the paper).
    Assumptions : 
     * Normalized gross moist stability = 0.2
     * pT/g=8000 kg/m2
    args:
     - x : 1D np.array, horizontal coordinate [m]
     - h : 1D np.array, surface height [m]
     - N : float, Brunt-Vaisala frequency [s^-1]
     - U : float, basic-state wind [m s^-1]
     - xi : float, damping rate [s^-1]
     - tauT : float, temperature adjustment time scale [s]
     - tauq : float, moisture adjustment time scale [s]
     - switch : int, set to a small number (e.g 10^-4 times the smallest wavenumber) to turn off the effect of Lq (equivalent to setting Lq=0, no precip relaxation).
     - zbot : float, lower edge of lower-free-troposphere [m]
     - ztop : float, upper edge of lower-free-troposphere [m]
    returns:
     - P' : array_like, precipitation in [mm day^-1]
     """
    pT_ov_g = 8e3 #mass of troposphere in kg/m2
    Lv = 2.5e6; cp=1004.; g = 9.81
    
    # array of heights in lower-free-troposphere
    z=np.arange(zbot,ztop+1,10.)
    
    # Horizontal wavenumber
    k=k_vector(len(x),x[1]-x[0]); k[0] = 1e-10 # dummy nonzero value
    
    # Fourier-transformed terrain
    h_hat = np.fft.fft(h); h_hat[0] = 0.

    # Relaxation length scale
    Lq=compute_Lq(5,U,tauq)
    
    # DSE and moisture lapse rates
    ds0dz = 300./g * N**2 * cp
    dq0dz = Lv * np.gradient(0.015 * np.exp(-z/2500),z)#.mean()
    
    chi = pT_ov_g * (ds0dz/tauT - dq0dz/tauq)/ Lv * 86400
    
    Pprimehat = 1j*k/(1j*k + switch*1/Lq)* h_hat * (chi * np.exp(1j* m_exponent_damped(k[:,None],N,U,xi,hydrostatic=hydrostatic)  *  z[None,:])).mean(axis=1) 
    
    P = np.real(np.fft.ifft(Pprimehat))
    return P

########################################################################################
############################  NONLINEAR MOUNTAIN WAVE THEORY  ##########################
########################################################################################
def hilbert_transform(signal):
    """Computes the hilbert transform of an arbitrary complex signal
    using Lilly&Klemp 1979's convention (which differs from scipy by a sign)
    """
    return -np.imag(hilbert(np.real(signal))) - 1j * np.imag(hilbert(np.imag(signal)))

def nonlinear_mountain_wave(x,z,h,N,U):
    """Computes vertical velocity in nonlinear mountain wave theory w/ constant N & U
    See RR Long, Some aspects of the flow of stratified fluids: I. A theoretical investigation. Tellus 5, 42–58 (1953).
    and D Lilly, J Klemp, The effects of terrain shape on nonlinear hydrostatic mountain waves. J. Fluid Mech. 95, 241–261 (1979).
    args:
     - x : 1D np.array, horizontal coordinate [m]
     - z : 1D np.array, vertical coordinate [m]
     - h : 1D np.array, surface height [m]
     - N : float, Brunt-Vaisala frequency [s^-1]
     - U : float, basic-state wind [m s^-1]
    returns:
     - w : 2D xr.DataArray, vertical velocity in [m s^-1]
    """
    f_L = hilbert_transform(h)
    H = h+1.j*f_L
    
    # scorer parameter
    l=N/U
    
    # Iterate equation 16 of Lilly&Klemp 1979
    niter=99
    for _ in range(niter):
        H = 1j*np.exp(-1j*l*h)*hilbert_transform(H*np.exp(1j*l*h))
    f=np.imag(H)

    # vertical displacement (eqn. 8 of Lilly&Klemp 1979)
    delta = h[:,None]*np.cos(l*(z[None,:]-h[:,None]))+f[:,None]*np.sin(l*(z[None,:]-h[:,None]))
    
    # vertical velocity
    w=U*np.gradient(delta,x,axis=0)
    
    # Temperature profile & pressure profile
    Tz = 300. + (300.*N**2/9.81 - 9.81/1004.)*z
    lnpz = - cumtrapz(9.81/(287.*Tz),z,initial=0.)
    
    return xr.DataArray(w,coords={'distance_from_mtn':x/1000,'z':('pressure',z/1000),'pressure':1000.*np.exp(lnpz)},dims=['distance_from_mtn','pressure'])