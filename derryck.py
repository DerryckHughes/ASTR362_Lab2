import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits 

def sixteen_to_twelve_correction(sixteen_bit_image):
    return sixteen_bit_image / 16 + 1

def mass_sixteen_to_twelve_correction(sixteen_bit_image_list):
    corrected_data = []
    for i in sixteen_bit_image_list:
        corrected_data.append(sixteen_to_twelve_correction(i))

    return np.array(corrected_data)

def photometry(x, y, x_mean, y_mean, x_stddev, y_stddev, cutout):
    '''
    Uses astropy Gaussian2D class to define a 2-dimensional gaussian fucntion

    Paramaters
    ----------
    x : array-like
    
        The x-values over which the gaussian will be applied

    y : array-like
    
        The x-values over which the gaussian will be applied

    x_mean : int
    
        x-center of the gaussian

    y_mean : int
    
        y-center of the gaussian

    x_stddev : int

        x-width of the gaussian

    y_stddev : int

        y-width of the gaussian

    Returns
    ----------
    gaussian(x,y)
        The 2D gaussian applied over the x-y matrix
    '''
    from astropy.modeling.functional_models import Gaussian2D
    gaussian = Gaussian2D(x_mean=x_mean, y_mean=y_mean, x_stddev=x_stddev, y_stddev=y_stddev)

    psf_convolve = gaussian(x,y) * cutout
    psf_flux = np.sum(psf_convolve)
    return gaussian(x,y), psf_convolve, psf_flux



def lin_least_square(x,y):
    "This is probably the ugliest way of doing this unfortunately"
    
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_sq_sum = 0
    xy_sum = 0
    for i in range(len(x)):
        x_sq_sum += x[i]**2
        xy_sum  += x[i] * y[i]

    delta = len(x) * x_sq_sum - x_sum **2
    a = ( x_sq_sum * y_sum - x_sum * xy_sum ) / delta
    b = ( len(x) * xy_sum - x_sum * y_sum ) / delta
    
    unc_y_sum = 0
    for i in range(len(x)):
        unc_y_sum += (y[i] - a -b * x[i] ) **2
    unc_y = np.sqrt( unc_y_sum / (len(x) - 2) )
    chi_sqr = unc_y_sum / unc_y **2

    unc_a = unc_y * np.sqrt( x_sq_sum / delta)
    unc_b = unc_y * np.sqrt( len(x) / delta)

    return a, b, unc_y, chi_sqr, unc_a, unc_b

def lin_least_square_weighted(x,y, uncs):
    '''
    Perform a linear least squars fit weighted with known uncertainties
    -----------
    Parameters:

    x:
        Array of independent variable values

    y: 
        Array of dependent variable values

    uncs: 
        Uncertainties in the x values

    -----------
    Returns:
    a, b:
        Coefficients for the linear equation y = ax + b
        
    unc_y:
        Uncertainty in the y_values

    chi_sqr:
        Float that is representative of a good fit, using the reduced chi squared is better though

    unc_a, unc_b:
        Uncertainties in a and b
    '''
    weights = []
    for i in uncs:
        weight = 1 / i **2
        weights.append(weight)
    weights = np.array(weights)
    
    x_sum = 0
    y_sum = 0
    x_sq_sum = 0
    xy_sum = 0
    weights_sum = np.sum(weights)
    for i in range(len(x)):
        x_sum += weights[i] * x[i]
        y_sum += weights[i] * y[i]
        x_sq_sum += weights[i] * x[i]**2
        xy_sum  += weights[i] * x[i] * y[i]

    delta = weights_sum * x_sq_sum - x_sum **2
    a = ( x_sq_sum * y_sum - x_sum * xy_sum ) / delta
    b = ( weights_sum * xy_sum - x_sum * y_sum ) / delta
    
    unc_y_sum = 0
    for i in range(len(x)):
        unc_y_sum += weights[i] * (y[i] - a - b * x[i] ) **2
    unc_y = np.sqrt( unc_y_sum / (len(x) - 2) )
    chi_sqr = unc_y_sum

    unc_a = np.sqrt( x_sq_sum / delta)
    unc_b = np.sqrt( weights_sum / delta)

    return a, b, unc_y, chi_sqr, unc_a, unc_b

def read_fits(fits_file):
    '''
    Opens a fits file in to an hdu to manipulate later
    ----------
    Parameters:

        fits_file
            The file to be opened
    ----------
    Returns:

        data
            The data from the fits file
        header
            The header of the fits file
    '''
    hdul = fits.open(fits_file)

    data = hdul[0].data
    header = hdul[0].header

    return data, header

def open_all_fits(list_of_files):
    ''' 
    Opens all fits files from a list of fits files and gets the data and the headers of each.
    -----------
    Parameters:
    
    list_of_files
        The list of fits files to be opened

    -----------
    Returns:

    data:
        A list of the data from all of the fits files

    headers:
        An list of headers from all of the fits files

    '''
    data, headers = [], []
    for i in list_of_files:
        data_i, header_i = read_fits(i)
        data.append(data_i)
        headers.append(header_i)

    return data, headers
    