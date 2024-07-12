import numpy as np
import xarray as xr
try:
    from xarray.ufuncs import cos, deg2rad
except ModuleNotFoundError:
    from numpy import cos
    def deg2rad(deg):
        return deg*np.pi/180
    
from scipy import special
from scipy.ndimage import gaussian_filter

###################################################################################################
###################################### BOXES, SELECTION ###########################################
###################################################################################################

def sel_box(da,box,lon='longitude',lat='latitude',lat_increase=False):
    """Select a range of latitudes and longitudes from a xr.DataArray
    args:
        - da : xr.Dataarray or xr.Dataset, from which to select
        - box : array-like of length 4 containing the longitude bounds and latitude bounds
                [lon_inf,lon_sup,lat_inf,lat_sup]
        - lon : str, name of the longitude coordinate. Default: 'longitude'
        - lat : str, name of the latitude coordinate. Default: 'latitude'
        - lat_increase : bool, whether the latitude coordinate is sorted in increasing order.
    returns :
        - xr.Dataarray or xr.Dataset, sliced from var
    """
    if lat_increase:
        return da.sel({lon:slice(box[0],box[1]),lat:slice(box[2],box[3])})
    else:
        return da.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})

def sel_months(ds,months):
    """Select a times in a xr.Dataset according to a list of months (that need not be contiguous)"""
    try: 
        return ds.sel(month = np.in1d( ds['month'], months))
    except KeyError:
        return ds.sel(time = np.in1d( ds['time.month'], months))

def sel_box_months(var,box,months,lon='longitude',lat='latitude',lat_increase=False):
    window = sel_box(var,box,lon,lat,lat_increase)
    window=sel_months(window,months)
    return window

def tilted_rect(grid,x1,y1,x2,y2,width,reverse=False,lon='longitude',lat='latitude'):
    """Creates a mask of a box that has a tilted rectangular shape (to outline rain bands)
    args :
        - grid : xr.Dataarray, defines the grid over which the mask will be created
        - x1 : float, longitude of the bottom-left corner of the rectangle
        - y1 : float, latitude of the bottom-left corner of the rectangle
        - x2 : float, longitude of the top-left corner of the rectangle
        - y2 : float, latitude of the top-left corner of the rectangle
        - width : float, width of the rectangle
        - reverse : bool, whether to flip the rectangle with respects to the ((x1,y1),(x2,y2)) axis
    returns :
        - xr.Dataarray, field of 0s and 1s defined on the grid that outline the tilted rectangle
    """
    x = grid[lon]
    y = grid[lat]
    if reverse:
        halfplane_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1) <=0
    else:
        halfplane_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1) >=0
    sc_prod = (x-x1)*(x2-x1)+(y-y1)*(y2-y1)
    halfplane_perp_up = sc_prod >= 0
    halfplane_perp_dn = (x-x2)*(x1-x2)+(y-y2)*(y1-y2) >= 0
    distance_across = np.sqrt(np.maximum(0.,(x-x1)**2+(y-y1)**2 - sc_prod**2/((x2-x1)**2+(y2-y1)**2)))
    return (halfplane_para*halfplane_perp_up*halfplane_perp_dn*(distance_across<width)).transpose(lat,lon)

def tilted_rect_distance(grid,x1,y1,x2,y2,distups1,distups2,lon='longitude',lat='latitude'):
    """Creates a mask of a box that has a tilted rectangular shape (to outline rain bands or regions upstream of them)
    This is a more general version of the above function. The idea is that we define an axis with two points. The rectangle
    will have two sides parallel to this axis, with length the distance between these two points. Their location and the 
    distance between them are defined by distups1 and distups2, which define distances upstream of the reference axis, 
    perpendicular to this axis.
    args :
        - grid : xr.Dataarray, defines the grid over which the mask will be created
        - x1 : float, longitude of the first point defining the reference axis
        - y1 : float, latitude of the first point defining the reference axis
        - x2 : float, longitude of the second point defining the reference axis
        - y2 : float, latitude of the second point defining the reference axis
        - distups1 : distance of the first side along of the reference axis. Negative distances allowed.
        - distups2 : distance of the second side along of the reference axis. Must be < distups1
    returns :
        - xr.Dataarray, field of 0s and 1s defined on the grid that outline the tilted rectangle
    """
    x = grid[lon]
    y = grid[lat]
    scalarprod_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1)
    scalarprod_perp_up = (x-x1)*(x2-x1)+(y-y1)*(y2-y1)
    scalarprod_perp_dn = (x-x2)*(x1-x2)+(y-y2)*(y1-y2)
    halfplane_para = scalarprod_para >=0
    halfplane_perp_up = scalarprod_perp_up >= 0
    halfplane_perp_dn = scalarprod_perp_dn >= 0
    distance_across = np.sqrt((x-x1)**2+(y-y1)**2 - scalarprod_perp_up**2/((x2-x1)**2+(y2-y1)**2))*np.sign(scalarprod_para)
    return (halfplane_perp_up*halfplane_perp_dn*(distance_across<distups1)*(distance_across>distups2)).transpose(lat,lon)



###########################################################################################################
########################################### SPATIAL FUNCTIONS #############################################
###########################################################################################################

def spatial_mean(ds,box=None,mask=None,lat='latitude',lon='longitude'):
    """Given a dataset with at least two spatial dimensions, compute a spatial mean within a specified region
    defined by a mask, inside a given box
        - variable = ND xarray.dataarray. Spatial dimensions names are given by the arguments 'lat' and 'lon'
        - mask = None, or 2D xarray.dataarray of 0s and 1s. Must have same grid and dimension names as 'variable'.
        - box = None, or list of four items, [lon1, lon2, lat1, lat2]
        - lat = str, name of the latitude coordinate
        - lon = str, name of the longitude coordinate
    """
    if type(ds)==int:
        return ds
    coslat = cos(deg2rad(ds[lat]))
    weight = coslat.expand_dims({lon:ds[lon]}) / coslat.mean(lat)
    if box :
        ds = ds.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
        weight = weight.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
        if mask is not None : 
            mask = mask.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
    if mask is None:
        mask=1
    
    mask = (mask*weight).where(~np.isnan(ds))
    ds   = mask*ds

    return ds.sum([lat,lon])/ mask.sum([lat,lon])

def ddx(F,lon='longitude'):
    """return zonal derivative in spherical coordinates"""
    coslat = np.cos(F.latitude*np.pi/180.)
    coslat += 1e-5*(1-1*(coslat>1e-5))
    m_per_degreelat = 6370*1e3*np.pi/180
    return F.differentiate(lon)/(m_per_degreelat*coslat)

def ddy(F,lat='latitude'):
    """return meridional derivative in spherical coordinates"""
    coslat = np.cos(F.latitude*np.pi/180.)
    coslat += 1e-5*(1-1*(coslat>1e-5))
    m_per_degreelat = 6370*1e3*np.pi/180
    return F.differentiate(lat)/m_per_degreelat

def divergence(Fx,Fy,lat='latitude',lon='longitude'):
    """return divergence in spherical coordinates"""
    coslat = np.cos(Fx[lat]*np.pi/180.)
    coslat += 1e-5*(1-1*(coslat>1e-5))
    m_per_degreelat = 6370*1e3*np.pi/180    
    return (Fx.differentiate(lon) + (Fy*coslat).differentiate(lat))/(m_per_degreelat*coslat)

def smooth(ds,sigma=2):
    """Gaussian smoothing of a xarray object"""
    return xr.apply_ufunc(gaussian_filter,ds,kwargs={"sigma":sigma})


###################################################################################################
#################################### BL & MISCELLANEOUS ###########################################
###################################################################################################

def crossslopeflow(u,v,angle):
    return (u*np.sin(angle*np.pi/180)+v*np.cos(angle*np.pi/180))

def compute_BL(thetaeB,thetaeL,thetaeLstar,wB=0.52):
    """Calculate the buoyancy surrogate B_L from thetaeB,thetaeL, and thetaeLstar.
    args:
        - thetaeB : array-like or float, boundary layer equivalent potential temperature in K
        - thetaeL : array-like or float, lower-free-tropospheric equivalent potential temperature in K
        - thetaeLstar : array-like or float, lower-free-tropospheric saturationequivalent potential 
          temperature in K
        - wB : importance of CAPE in setting plume buoyancy. See Ahmed, Adames & Neelin (JAS 2020).
    returns:
        - B_L, xr.DataArray, buoyancy in m/s^2
    """
    g=9.81
    wL = 1-wB
    thetae0 = 340
    capeL   = (thetaeB/thetaeLstar - 1)*thetae0
    subsatL = (1 - thetaeL/thetaeLstar)*thetae0
    BL = g/thetae0*(wB*capeL-wL*subsatL) # Did not include the KappaL term to conform with AAN2020
    return BL

###################################################################################################
##################################### STATISTICAL TOOLS ###########################################
###################################################################################################
def linregress_xr(x,y,dim='time'):
    """Linear regression of variable y on variable x along a specified dimension. 
    Wrote this vectorized function which is much faster than looping over a pre-defined function.
    args:
        - x : xr.DataArray, usually a timeseries 
        - y : xr.DataArray, can have any number of dimensions
        - dim : str, the dimension to perform the regression on
    returns:
        - xr.DataArray containing all the dimension in y except dim. Contains the following variables:
          intercept, slope, rsquared, pvalue (2-sided t-test on whether the slope is =/= 0),
          slope_sterr (standard error of slope), slope_lower and slope_upper (95% CI on slope)
    """
    nt = len(x[dim])
    assert nt == len(y[dim])
    ssxm = nt * x.var(dim=dim,ddof=0)
    ssym = nt * y.var(dim=dim,ddof=0)
    ssxym = nt * xr.cov(x, y, dim=dim,ddof=0)
    
    r2 = ssxym**2 / (ssxm * ssym)
    slope = ssxym / ssxm
    itcpt = y.mean(dim=dim) - slope * x.mean(dim=dim)
    
    df = nt - 2  # Number of degrees of freedom
    se = np.sqrt( (ssxm * ssym - ssxym**2) / ssxm**2 / df)  # Standard error of slope
    
    TINY = 1e-20
    t = slope/(se + TINY) # Calculate the t-statistic
    pval = special.stdtr(df, -np.abs(t)) * 2 * y.isel({dim: 0}) ** 0  # Calculate the two-tailed p-value
    
    alpha = 0.05  # Significance level
    t_crit = special.stdtrit(df,1 - alpha/2) # Calculate appropriate quantile of the t distribution
    slope_lower = slope - t_crit * se  # Lower bound of confidence interval
    slope_upper = slope + t_crit * se  # Upper bound of confidence interval

    return xr.merge([itcpt.rename('intercept'), 
                     slope.rename('slope'), 
                     r2.rename('rsquared'), 
                     pval.rename('pvalue'), 
                     se.rename('slope_sterr'), 
                     slope_lower.rename('slope_lower'), 
                     slope_upper.rename('slope_upper')])




def fdr(pvalues,alpha):
    """Given an array of p-values and a false discovery rate (FDR) level, 
    return the p-value threshold that defines significance according to this FDR level.
    See Wilks (2016) for more info.
    """
    sortidx = np.argsort(pvalues)
    psorted = pvalues[sortidx]
    psorted[np.isnan(psorted)]=1
    nval = len(pvalues)
    ifdr = np.argmax((psorted < alpha*np.arange(1,nval+1)/nval)[::-1])
    if ifdr == 0 and psorted[-1]>= alpha:
        ifdr=nval-1
    ifdr = nval - ifdr - 1
    return sortidx[:ifdr]

def fdr_xr_2d(pvalues,alpha):
    """Given an map of p-values and a false discovery rate (FDR) level, 
    return the mask that defines significance according to this FDR level.
    See Wilks (2016) for more info.
    """    
    pvalues=np.array(pvalues)
    assert len(pvalues.shape)==2
    ntot = pvalues.shape[0]*pvalues.shape[1]
    idxs_1d = fdr(pvalues.reshape(-1),alpha)
    flags = np.zeros(ntot)
    flags[idxs_1d] = 1
    return flags.reshape(pvalues.shape)*pvalues**0

def generate_sample(timeseries, blocksize=1, nbootstrap=10000):
    """
    Generate bootstrap samples from a time series using the block bootstrap method.

    Parameters:
    timeseries (numpy.ndarray): The time series to generate bootstrap samples from.
    blocksize (int, optional): The size of the blocks to use for bootstrapping. Default is 1.
    nbootstrap (int, optional): The number of bootstrap samples to generate. Default is 10000.

    Returns:
    numpy.ndarray: An array of bootstrap samples, with shape (nbootstrap, Nt) where Nt is the length of the original time series.
    """
    Nt = len(timeseries)  # Get the length of the time series

    nsamples_per_bootstrap = Nt // blocksize  # Calculate the number of samples to include in each bootstrap sample
    nsamples = nbootstrap * nsamples_per_bootstrap  # Calculate the total number of samples to generate

    idxs = np.random.choice(Nt - blocksize, nsamples, replace=True)  # Generate random indices for the blocks
    idx_blocks = np.broadcast_to(idxs, (blocksize, nsamples)).T + np.arange(blocksize)  # Create an array of block indices
    bootstrap_samples = timeseries[idx_blocks].reshape(nbootstrap, nsamples_per_bootstrap * blocksize)  # Generate the bootstrap samples

    return bootstrap_samples

def generate_joint_sample(timeseries_list, blocksize=1, nbootstrap=10000):
    """
    Generate bootstrap samples from several timeseries using the block bootstrap method.

    Parameters:
    timeseries_list (numpy.ndarray): The array of time series to generate bootstrap samples from. Shape must be [Ntimeseries, Ntimes]
    blocksize (int, optional): The size of the blocks to use for bootstrapping. Default is 1.
    nbootstrap (int, optional): The number of bootstrap samples to generate. Default is 10000.

    Returns:
    numpy.ndarray: An array of bootstrap samples, with shape (nbootstrap, Nt) where Nt is the length of the original time series.
    """
    Ntimeseries, Nt = timeseries_list.shape 

    nsamples_per_bootstrap = Nt // blocksize  # Calculate the number of samples to include in each bootstrap sample
    nsamples = nbootstrap * nsamples_per_bootstrap  # Calculate the total number of samples to generate

    idxs = np.random.choice(Nt - blocksize, nsamples, replace=True)  # Generate random indices for the blocks
    idx_blocks = np.broadcast_to(idxs, (blocksize, nsamples)).T + np.arange(blocksize)  # Create an array of block indices
    bootstrap_samples = timeseries_list[:,idx_blocks].reshape(Ntimeseries,nbootstrap, nsamples_per_bootstrap * blocksize)  # Generate the bootstrap samples

    return bootstrap_samples