import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from numpy import savetxt

class particle:
    """particle: class that define a particle
    """
    def __init__(self, n_atom=1, rads_atom=[1,], size=10., crds = None, size_grid=32, size_image=25):
        """
        Arguments:
        ---------
        - n_atom (integer {default:1}): 
            number of atoms in particle
        - rads_atom (list {default:[1,]}): 
            list of atoms radii (in Angstroem)
        - size (float {default:10.}): 
            particle radius (in Angstroem)
        - size_grid (integer {default:32}):
            number of voxels along each grid direction
        - size_image (float {default: 25.}):
            size of the grid (in Angstroem)
        """
        self.n_atom = n_atom
        self.rads_atom = rads_atom
        self.size = size
        if crds is None:
            self.crds = size * np.random.rand(n_atom, 3)
        else:
            self.crds = crds
        self.crds -= np.mean(self.crds, axis=0)
        self.rads = rads_atom
        self.image = None
        self.size_grid = size_grid
        self.size_image = size_image
        self.volume = None
        self.last_rot = [0,0,0]
        
    def rotate(self, rotvec=None):
        """
        Rotating the particle with the rotation vector.
        If no rotation vector is given, rotates randomly.
        """
        if rotvec is None:
            rotation =  R.random()
        else:
            rotation = R.from_rotvec(rotvec)
        
        abs_rotation = rotation * R.from_rotvec(self.last_rot)
        self.last_rot = abs_rotation.as_rotvec()
        
        for i in range(self.n_atom):
            self.crds[i] = rotation.apply(self.crds[i])
            
        self.create_image()
    
    def create_image(self):
        """
        Return the (real-space) projection of the particle along the z axis
        """
        image = np.zeros((self.size_grid,self.size_grid))
        for i in range(self.size_grid):
            posx = -self.size_image/2 + i*self.size_image/(self.size_grid-1)
            for j in range(self.size_grid):
                posy = -self.size_image/2 + j*self.size_image/(self.size_grid-1)
                
                for iat in range(self.n_atom):
                    dx = self.crds[iat,0] - posx
                    dy = self.crds[iat,1] - posy
                    dist2 = dx*dx + dy*dy
                    if dist2 <= self.size**2:
                        #integrating along z axis
                        image[i,j] += np.sqrt(2*np.pi*self.size_image)*self.rads[iat]*np.exp(-0.5*dist2/self.rads[iat]**2)
                        
        self.image = image
    
    def show(self):
        if self.has_image:
            plt.imshow(self.image, cmap='Greys') 
            
    def create_map(self):
        """
        return a 3d array of the particle
        """
        volume = np.zeros((self.size_grid,self.size_grid,self.size_grid))
        for i in range(self.size_grid):
            posx = -self.size_image/2 + i*self.size_image/(self.size_grid-1)
            for j in range(self.size_grid):
                posy = -self.size_image/2 + j*self.size_image/(self.size_grid-1)
                for k in range(self.size_grid):
                    posz = -self.size_image/2 + k*self.size_image/(self.size_grid-1)
                    
                    for iat in range(self.n_atom):
                        dx = self.crds[iat,0] - posx
                        dy = self.crds[iat,1] - posy
                        dz = self.crds[iat,2] - posz
                        dist2  = dx*dx + dy*dy + dz*dz
                        if(dist2 <= 100):
                            volume[i,j,k] += np.exp(-0.5*dist2/self.rads[iat]**2)
        self.volume = volume


def generate_dataset(particle, n_projections=1, 
                     reldir='.', keyword='particle',
                     print_frequency=100, 
                     output_map=True, output_xyz=True):
    """ generate_dataset: generate a particle image dataset from particle object
    
    Arguments:
    ---------
    - particle (class):
        particle to generate dataset from
    - n_projections (integer {default:1}):
        number of projection images to generate from particle
    - reldir (string {default: '.'}):
        relative path to directory where to save dataset
    - keyword (string {default: 'particle'}):
        keyword used to define dataset 
    - print_frequency (integer {default: 100}):
        print some information every print_frequency particles
    - output_map (bool {default: True}):
        whether the map is written to numpy array
    - output_xyz (bool {default: True}):
        whether the crds and rad of the particle atoms are written to numpy array.
    
    Outputs:
    -------
    - dataset array of shape (n_projections, size_grid, size_grid)
    - metadataset array of shape (n_projections, 5):
        for each projection image, contains the following information:
        . quaternion 
        . image size
        . number of atoms in particle
    - files: dataset, metadataset, map (optional), xyz (optional)
    """
    
    dataset     = np.zeros((n_projections, particle.size_grid, particle.size_grid))
    metadataset = np.zeros((n_projections, 5))

    if output_map: #needs to be called before rotating the particle
        particle.create_map() 
        np.save(f'{reldir}/{keyword}_map', particle.volume)
    
    for i in np.arange(n_projections):
        particle.rotate()
        dataset[i,...]   = particle.image
        metadataset[i,0:3] = np.array(particle.last_rot)
        metadataset[i,3] = particle.size_image
        metadataset[i,4] = particle.n_atom
        if (i+1)%print_frequency == 0:
            print(f'{str(i+1)} images were created')
    
    np.save(f'{reldir}/{keyword}_data', dataset)
    np.save(f'{reldir}/{keyword}_meta', metadataset)
    
    if output_xyz:
        xyz = np.array(particle.crds)
        rad = np.array(particle.rads)
        array = np.zeros((xyz.shape[0], 4))
        array[:,0:3] = xyz
        array[:,3]   = rad
        np.save(f'{reldir}/{keyword}_xyz', array)

    return dataset, metadataset
