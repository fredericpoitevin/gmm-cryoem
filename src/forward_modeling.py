import numpy as np
#from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator
#from scipy.interpolate import griddata

from scipy.interpolate import griddata, RegularGridInterpolator
def slice_volume(vol, Rotation, method='linear'):
    """
    Take a slice out of volume
    """
    N = vol.shape[0]
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)
    interpolating_function = RegularGridInterpolator(points=[x,y,z],values=vol,bounds_error = False, fill_value=0, method=method)

    grid_x, grid_y = np.meshgrid(x,y, indexing='ij')
    zeros = np.zeros((grid_x.shape))
    points = np.stack([grid_x.ravel(),grid_y.ravel(), zeros.ravel()],1)
    rot_points = Rotation.apply(points)

    image = interpolating_function(rot_points)
    return image.reshape(N,N)

#def project_volume(vol, Rotation):
#    """
#    Given a real-space density volume, retrieve real-space projection after Rotation
#    """
#    vol_ft = np.fft.fftn(vol)
#    vol_shift = np.fft.fftshift(vol_ft)
#    im_ft = slice_volume(vol_shift, Rotation)
#    im = np.fft.ifftshift(im_ft)
#    projection = np.real(np.fft.ifft2(im))
    
#    return projection

def rotate_volume(vol, Rotation):
    """
    This function rotate volume by Rotation
    """
    N = vol.shape[0]
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)
    interpolating_function = RegularGridInterpolator(points=[x,y,z],values=vol,bounds_error = False, fill_value=0)

    grid_x, grid_y,grid_z  = np.meshgrid(x,y,z, indexing='ij')
    points = np.stack([grid_x.ravel(),grid_y.ravel(), grid_z.ravel()],1)
    rot_points = Rotation.apply(points)

    volume_ = interpolating_function(rot_points)
    return volume_.reshape((N,N,N), order='C')

def take_slice(vol, phase_amp_mode = False):
    n=vol.shape[0]
    if phase_amp_mode:
        return (np.abs(vol[:,:,n//2])+np.abs(vol[:,:,n//2+1]))/2*np.exp(1j*(np.angle(vol[:,:,n//2])+np.angle(vol[:,:,n//2+1]))/2)        
    else:
        return (vol[:,:,n//2]+vol[:,:,n//2+1])/2

def project_volume(vol, Rotation):
    """
    Given a real-space density volume, retrieve real-space projection after Rotation
    """
    rot_vol = rotate_volume(vol, Rotation)
    rot_vol_ft = np.fft.fftn(rot_vol)
    rot_vol_shift = np.fft.fftshift(rot_vol_ft)
    im_ft = take_slice(rot_vol_shift)
    im = np.fft.ifftshift(im_ft)
    projection = np.real(np.fft.ifft2(im))
    
    return 50*projection #to correct scale problem
