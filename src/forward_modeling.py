import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata, RegularGridInterpolator

def slice_volume(vol, Rotation, method='linear', angle_abs_mode = False):
    """
    Take a slice out of volume
    """
    N = vol.shape[0]
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    z = np.linspace(-1,1,N)
    
    grid_x, grid_y = np.meshgrid(x,y, indexing='ij')
    zeros = np.zeros((grid_x.shape))
    points = np.stack([grid_x.ravel(),grid_y.ravel(), zeros.ravel()],1)
    rot_points = Rotation.apply(points)
    
    if not(angle_abs_mode):
        interpolating_function = RegularGridInterpolator(points=[x,y,z],values=vol,bounds_error = False, fill_value=0, method=method)
        image = interpolating_function(rot_points)

    else:
        angle_interpolating_function = RegularGridInterpolator(points=[x,y,z],values=np.angle(vol),bounds_error = False, method=method)
        abs_interpolating_function = RegularGridInterpolator(points=[x,y,z],values=np.abs(vol),bounds_error = False, fill_value=0, method=method)
        image = abs_interpolating_function(rot_points)*np.exp(1j*abs_interpolating_function(rot_points))

    return image.reshape(N,N)

def project_volume_bis(vol, Rotation, angle_abs_mode = False, mask_radius=15):
    """
    Given a real-space density volume, retrieve real-space projection after Rotation
    """
    mask_vol = create_circular_mask(vol.shape[0], radius = mask_radius, is_volume=True)
    mask_slice = create_circular_mask(vol.shape[0], radius = mask_radius, is_volume=False)

    vol_ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol)))
    vol_ft[~mask_vol]=0
    im_ft = slice_volume(vol_ft, Rotation, angle_abs_mode = angle_abs_mode)
    im_ft[~mask_slice]=0
    im = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(im_ft))))
    #im[~mask_slice]=0
    return im

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
    projection = np.sum(rot_vol, axis=2)
    return projection




######################################################################################################################################################################################################################################################################################################################## BACKPROJECTION ##############################################################################################################################################################################################################################################################################################################################

def create_circular_mask(N, center=None, radius=None, is_volume=False):
    if radius is None: 
        radius = int(N/2)
    if center is None: # use the middle of the image
        if (is_volume):
            center = (int(N/2), int(N/2), int(N/2))
        else:
            center = (int(N/2), int(N/2))

    if (is_volume):        
        Z, Y, X = np.ogrid[:N, :N, :N]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    if not(is_volume):        
        Y, X = np.ogrid[:N, :N]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        
    mask = dist_from_center <= radius
    return mask

def backprojection(images, orientations):
    
    N=images[0].shape[0]
    vol = np.zeros((N,N,N),  dtype=np.complex)
    counts = np.zeros((N,N,N))
    #mask1 = create_circular_mask(N, radius = 10)
    
    for i in range(len(images)):
        #adding the contribution of each slices/images
        rot = R.from_rotvec(-orientations[i])
        image_i=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(images[i])))
        #image_i[~mask1]=0
        vol, counts = add_slice(vol, counts, image_i, rot)

    counts[counts == 0] = 1
    mask = create_circular_mask(N, radius = 10)
    #return vol/counts
    vol = vol/counts
    #vol[~mask] = 0
    
    return np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(vol))))
   

def lattice_create_mask(lattice, radius=1., mode='cubic'):
    """
    Mask for vectors with real cordinates.
    We'll use this function to work with the unit ball instead of the whole volume.
    """
    if mode == 'circular':
        dist_from_center = np.sqrt((lattice[:,0])**2 + (lattice[:,1])**2 + (lattice[:,2])**2)
        mask = (dist_from_center < radius)
    elif mode == 'cubic':
        mask = ((lattice[:,0]**2<1.) & (lattice[:,1]**2<1.) & (lattice[:,2]**2<1.))
    return mask


def add_slice(vol, counts, image, rot, prob = 1):
    """
    This function adds an image to the volume, by updating the values of the pixels arround the slice.
    """
    N = vol.shape[0]
    image=image.reshape((N*N,1))
    d2 = (N-1)/2
    
    #building rotated lattice
    x = np.linspace(-1,1,N)
    y = np.linspace(-1,1,N)
    grid_x, grid_y = np.meshgrid(x,y, indexing='ij')
    zeros = np.zeros((grid_x.shape))
    points = np.stack([grid_x.ravel(),grid_y.ravel(), zeros.ravel()],1)
    rot_points = rot.apply(points)
    
    #coordinates of the adjacent voxels
    points_f = np.floor((rot_points+1)*d2)/d2-1
    points_c = np.ceil((rot_points+1)*d2)/d2-1

    #By mixing the floor values and the ceil values, we can browse all adjacent voxels
    #we only keep the points that are inside the unit ball
    xf, yf, zf = np.split(points_f,3,1)
    xc, yc, zc = np.split(points_c,3,1)
    
    def add_for_corner(xi,yi,zi):
        """
        This auxilliary function add the contribution of the slice to a set of adjacent voxels
        """
        mask_i = lattice_create_mask(np.stack((xi,yi,zi), axis=1)[:,:,0])
        dist = (np.stack((xi,yi,zi), axis=1)[:,:,0] - rot_points[:])*d2
        dist = dist[mask_i]
        w = 1 - np.sqrt(np.sum(np.power(dist, 2), axis=1))
        w[w<0]=0
        w=w.reshape((-1, 1))
        
        #mask_z = (image[mask_i]<0.1) #why not
        #w[mask_z]=0   #why not
        vol[np.around(xi[mask_i]*d2+d2).astype(int),np.around(yi[mask_i]*d2+d2).astype(int),np.around(zi[mask_i]*d2+d2).astype(int)] += (w*image[(mask_i)]) * prob
        counts[np.around(xi[mask_i]*d2+d2).astype(int),np.around(yi[mask_i]*d2+d2).astype(int),np.around(zi[mask_i]*d2+d2).astype(int)] += w * prob
    
    #calling the auxilliary function for all adjacent voxels
    add_for_corner(xf,yf,zf)
    add_for_corner(xc,yf,zf)
    add_for_corner(xf,yc,zf)
    add_for_corner(xf,yf,zc)
    add_for_corner(xc,yc,zf)
    add_for_corner(xf,yc,zc)
    add_for_corner(xc,yf,zc)
    add_for_corner(xc,yc,zc)
    
    return vol, counts




