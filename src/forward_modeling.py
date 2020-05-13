import numpy as np
#from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator
#from scipy.interpolate import griddata

def slice_volume(vol, Rotation):
    """
    Take a slice out of volume
    """
    print(Rotation)

    N = vol.shape[0]
    image = np.zeros((N,N), dtype=np.complex_)
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)
    interpolating_function = RegularGridInterpolator((x,y,z), vol)

    for i in range(N):
        x = -1 + i*2/(N-1)
        for j in range(N):
            y = -1 + j*2/(N-1)
            
            vect = np.array((x,y,0))
            vect_ = Rotation.apply(vect)
            if np.max(np.abs(vect_))<=1:
                image[i,j] += interpolating_function(vect_)

    return image

def project_volume(vol, Rotation):
    """
    Given a real-space density volume, retrieve real-space projection after Rotation
    """
    vol_ft = np.fft.fftn(vol)
    vol_shift = np.fft.fftshift(vol_ft)
    im_ft = slice_volume(vol_shift, Rotation)
    im = np.fft.ifftshift(im_ft)
    projection = np.real(np.fft.ifft2(im))
    
    return projection
