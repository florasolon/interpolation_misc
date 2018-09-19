
# coding: utf-8
# In[ ]:
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib import cm


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    
    return my_cmap_r

def plot_models_6sub(distx,disty,profz,x,y,z,x0,y0,z0,rho_o,rho_c,vmin=1, vmax=1000):
    '''
    Plot real and interpolated models in 6 subplots.
    
    Input
    profz: scalar - depth.
    distx: scalar - distance in x direction.
    disty: scalar - distance in y direction.
    x: np array - coordinates x of the model.
    y: np array - coordinates y of the model.
    z: np array - coordinates z of the model.
    x0: np array - coordinates x of the model.
    y0: np array - coordinates y of the interpolated model.
    z0: np array - coordinates z of the interpolated model.
    rho_o: np matrix - observed model to be ploted.
    rho_c: np matrix - calculated model to be ploted.
    
    output
    plot
    '''
    
    x_plot = np.argmin(np.abs(x-distx))
    y_plot = np.argmin(np.abs(y-disty))
    z_plot = np.argmin(np.abs(z-profz))
    x0_plot = np.argmin(np.abs(x0-distx))
    y0_plot = np.argmin(np.abs(y0-disty))
    z0_plot = np.argmin(np.abs(z0-profz))
    
    mx,my = np.meshgrid(y,x)
    mx1,my1 = np.meshgrid(z,y)
    mx2,my2 = np.meshgrid(z,x)
    
    m0x,m0y = np.meshgrid(y0,x0)
    m0x1,m0y1 = np.meshgrid(z0,y0)
    m0x2,m0y2 = np.meshgrid(z0,x0)
    
    cmap = plt.cm.jet
    cmap_r = reverse_colourmap(cmap)
    
    
    plt.figure(figsize=(14,10))

    plt.subplot(3,2,1)
    plt.pcolor(mx,my,rho_o[:,:,z_plot],cmap=cmap_r,norm = LogNorm(vmin, vmax))
    plt.colorbar()
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('Observed')
    
    plt.subplot(3,2,2)
    plt.pcolor(m0x,m0y,rho_c[:,:,z0_plot],cmap=cmap_r,norm = LogNorm(vmin, vmax))
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Calculated')

    plt.subplot(3,2,3)
    plt.pcolor(my1,mx1,rho_o[x_plot,:,:],cmap=cmap_r,norm = LogNorm(vmin, vmax))
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.ylim(np.max(z),np.min(z))
    plt.colorbar()

    plt.subplot(3,2,4)
    plt.pcolor(m0y1,m0x1,rho_c[x0_plot,:,:],cmap=cmap_r,norm = LogNorm(vmin, vmax))
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.ylim(np.max(z0),np.min(z0))
    plt.colorbar()

    plt.subplot(3,2,5)
    plt.pcolor(my2,mx2,rho_o[:,y_plot,:],cmap=cmap_r,norm = LogNorm(vmin, vmax))
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(np.max(z),np.min(z))
    plt.colorbar()

    plt.subplot(3,2,6)
    plt.pcolor(m0y2,m0x2,rho_c[:,y0_plot,:],cmap=cmap_r,norm = LogNorm(vmin, vmax))
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(np.max(z0),np.min(z0))
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
def plot_function_6sub(distx,disty,profz,x,y,z,x0,y0,z0,rho_o,rho_c):
    '''
    Plot real and interpolated models in 6 subplots.
    
    Input
    profz: scalar - depth.
    distx: scalar - distance in x direction.
    disty: scalar - distance in y direction.
    x: np array - coordinates x of the model.
    y: np array - coordinates y of the model.
    z: np array - coordinates z of the model.
    x0: np array - coordinates x of the model.
    y0: np array - coordinates y of the interpolated model.
    z0: np array - coordinates z of the interpolated model.
    rho_o: np matrix - observed model to be ploted.
    rho_c: np matrix - calculated model to be ploted.
    
    output
    plot
    '''
    
    x_plot = np.argmin(np.abs(x-distx))
    y_plot = np.argmin(np.abs(y-disty))
    z_plot = np.argmin(np.abs(z-profz))
    x0_plot = np.argmin(np.abs(x0-distx))
    y0_plot = np.argmin(np.abs(y0-disty))
    z0_plot = np.argmin(np.abs(z0-profz))
    
    mx,my = np.meshgrid(y,x)
    mx1,my1 = np.meshgrid(z,y)
    mx2,my2 = np.meshgrid(z,x)
    
    m0x,m0y = np.meshgrid(y0,x0)
    m0x1,m0y1 = np.meshgrid(z0,y0)
    m0x2,m0y2 = np.meshgrid(z0,x0)
    
    plt.figure(figsize=(14,10))

    plt.subplot(3,2,1)
    plt.pcolor(mx,my,rho_o[:,:,z_plot])
    plt.colorbar()
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('Observed')
    
    plt.subplot(3,2,2)
    plt.pcolor(m0x,m0y,rho_c[:,:,z0_plot])
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Calculated')

    plt.subplot(3,2,3)
    plt.pcolor(my1,mx1,rho_o[x_plot,:,:])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.ylim(np.max(z),np.min(z))
    plt.colorbar()

    plt.subplot(3,2,4)
    plt.pcolor(m0y1,m0x1,rho_c[x0_plot,:,:])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.ylim(np.max(z0),np.min(z0))
    plt.colorbar()

    plt.subplot(3,2,5)
    plt.pcolor(my2,mx2,rho_o[:,y_plot,:])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(np.max(z),np.min(z))
    plt.colorbar()

    plt.subplot(3,2,6)
    plt.pcolor(m0y2,m0x2,rho_c[:,y0_plot,:])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(np.max(z0),np.min(z0))
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
def plot_function_interp(distx,disty,profz,x,y,z,x0,y0,z0,rho_o,rho_c):
    '''
    Plot real and interpolated models.
    
    Input
    profz: scalar - depth.
    distx: scalar - distance in x direction.
    disty: scalar - distance in y direction.
    x: np array - coordinates x of the model.
    y: np array - coordinates y of the model.
    z: np array - coordinates z of the model.
    x0: np array - coordinates x of the model.
    y0: np array - coordinates y of the interpolated model.
    z0: np array - coordinates z of the interpolated model.
    rho_o: np matrix - observed model to be ploted.
    rho_c: np matrix - calculated model to be ploted.
    
    output
    plot
    '''
    
    x_plot = np.argmin(np.abs(x-distx))
    y_plot = np.argmin(np.abs(y-disty))
    z_plot = np.argmin(np.abs(z-profz))
    x0_plot = np.argmin(np.abs(x0-distx))
    y0_plot = np.argmin(np.abs(y0-disty))
    z0_plot = np.argmin(np.abs(z0-profz))
    
    mx,my = np.meshgrid(y,x)
    mx1,my1 = np.meshgrid(z,y)
    mx2,my2 = np.meshgrid(z,x)
    
    m0x,m0y = np.meshgrid(y0,x0)
    m0x1,m0y1 = np.meshgrid(z0,y0)
    m0x2,m0y2 = np.meshgrid(z0,x0)
    
    plt.figure(figsize=(14,10))

    plt.subplot(3,1,1)
    plt.pcolor(mx,my,rho_o[:,:,z_plot])
    plt.colorbar()
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('Observed')
    
    plt.subplot(3,2,2)
    plt.pcolor(m0x,m0y,rho_c[:,:,z0_plot])
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Calculated')

    plt.subplot(3,2,3)
    plt.pcolor(my1,mx1,rho_o[x_plot,:,:])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.ylim(np.max(z),np.min(z))
    plt.colorbar()

    plt.subplot(3,2,4)
    plt.pcolor(m0y1,m0x1,rho_c[x0_plot,:,:])
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.ylim(np.max(z0),np.min(z0))
    plt.colorbar()

    plt.subplot(3,2,5)
    plt.pcolor(my2,mx2,rho_o[:,y_plot,:])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(np.max(z),np.min(z))
    plt.colorbar()

    plt.subplot(3,2,6)
    plt.pcolor(m0y2,m0x2,rho_c[:,y0_plot,:])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.ylim(np.max(z0),np.min(z0))
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def select_area(xmin,xmax,ymin,ymax,zmin,zmax,x,y,z,rho):
    '''
    Returns the coordinate arrays x,y,z 
    '''
    assert (xmin >=0), 'coordinate must be higher than zero'
    assert (xmax >=0), 'coordinate must be higher than zero'
    assert (ymin >=0), 'coordinate must be higher than zero'
    assert (ymax >=0), 'coordinate must be higher than zero'
    assert (zmin >=0), 'coordinate must be higher than zero'
    assert (zmax >=0), 'coordinate must be higher than zero'
    
    i_xmin = np.argmin(np.abs(x-xmin))
    i_xmax = np.argmin(np.abs(x-xmax))
    
    i_ymin = np.argmin(np.abs(y-ymin))
    i_ymax = np.argmin(np.abs(y-ymax))
    
    i_zmin = np.argmin(np.abs(z-zmin))
    i_zmax = np.argmin(np.abs(z-zmax))

    x_ = x[i_xmin:i_xmax]
    y_ = y[i_ymin:i_ymax]
    z_ = z[i_zmin:i_zmax]

    X,Y,Z = np.meshgrid(x_,y_,z_)

    x_1 = np.ravel(X)
    y_1 = np.ravel(Y)
    z_1 = np.ravel(Z)
    
    rho_1 = rho[i_xmin:i_xmax,i_ymin:i_ymax,i_zmin:i_zmax]
    
    return x_1,y_1,z_1,x_,y_,z_,rho_1

def select_area_latlon(xmin,xmax,ymin,ymax,zmin,zmax,x,y,z,rho):
    '''
    Returns the coordinate arrays x,y,z 
    '''
        
    i_xmin = np.argmin(np.abs(x-xmin))
    i_xmax = np.argmin(np.abs(x-xmax))
    
    i_ymin = np.argmin(np.abs(y-ymin))
    i_ymax = np.argmin(np.abs(y-ymax))
    
    i_zmin = np.argmin(np.abs(z-zmin))
    i_zmax = np.argmin(np.abs(z-zmax))

    x_ = x[i_xmin:i_xmax]
    y_ = y[i_ymin:i_ymax]
    z_ = z[i_zmin:i_zmax]

    X,Y,Z = np.meshgrid(x_,y_,z_)

    x_1 = np.ravel(X)
    y_1 = np.ravel(Y)
    z_1 = np.ravel(Z)
    
    rho_1 = rho[i_xmin:i_xmax,i_ymin:i_ymax,i_zmin:i_zmax]
    
    return x_1,y_1,z_1,x_,y_,z_,rho_1

def calc_gij(xi,yi,zi,xj,yj,zj,t=None):
    '''
    Returns the coefficients for spline interpolation using Green’s functions.
    '''
    assert isinstance(1.*xi, float), 'xi must be a float' 
    assert isinstance(1.*yi, float), 'yi must be a float'
    assert isinstance(1.*zi, float), 'zi must be a float'
    
    assert isinstance(1.*xj, float), 'xj must be a float' 
    assert isinstance(1.*yj, float), 'yj must be a float'
    assert isinstance(1.*zj, float), 'zj must be a float'
    
    if t is None:
        gij = np.sqrt(((xi-xj)*(xi-xj))+((yi-yj)*(yi-yj))+((zi-zj)*(zi-zj)))
        
    else:
        assert (t>=0) & (t<1), 't must lie between 0 and 1'
        p = np.sqrt(t/(1-t))
        r = np.sqrt(((xi-xj)*(xi-xj))+((yi-yj)*(yi-yj))+((zi-zj)*(zi-zj)))
        gij = ((1/(p*r))*(np.exp(-p*r)-1))+1
    
    return gij

def calc_A(x,y,z,xc,yc,zc,t=None):
    '''
    Returns the sensibility matrix for spline interpolation.
    
    input
    x: np array - coordinates x of the data.
    y: np array - coordinates y of the data.
    z: np array - coordinates z of the data.
    
    xc: np array - coordinates x of the control points to calculate green function for spline interpolation.
    yc: np array - coordinates y of the control points to calculate green function for spline interpolation.
    zc: np array - coordinates z of the control points to calculate green function for spline interpolation.
    
    output
    A: np array 
    '''
    assert (x.size == y.size == z.size), 'Sizes of x, y and z must be equal '
    assert (xc.size == yc.size == zc.size), 'Sizes of xc, yc and zc must be equal '
    
    N = x.size
    M = xc.size

    A = np.zeros((N,M))

    for i,(xi,yi,zi) in enumerate(zip(x,y,z)):
        for j,(xj,yj,zj) in enumerate(zip(xc,yc,zc)):
            A[i,j] = calc_gij(xi,yi,zi,xj,yj,zj,t)
            
    return A

def calc_g_column1(x,y,z,xj,yj,zj,t=None):
    '''
    Returns the coefficients for spline interpolation using Green’s functions.
    '''
    assert isinstance(1.*x, np.ndarray), 'x must be a list' 
    assert isinstance(1.*y, np.ndarray), 'y must be a list' 
    assert isinstance(1.*z, np.ndarray), 'z must be a list' 
    assert (x.size == y.size == z.size), 'Sizes of x, y and z must be equal '   
    assert isinstance(1.*xj, float), 'xj must be a float' 
    assert isinstance(1.*yj, float), 'yj must be a float'
    assert isinstance(1.*zj, float), 'zj must be a float'
    
    if t is None:
        g = np.sqrt(((x-xj)*(x-xj))+((y-yj)*(y-yj))+((z-zj)*(z-zj)))
        
    else:
        assert (t>=0) & (t<1), 't must lie between 0 and 1'
        p = np.sqrt(t/(1-t))
        r = np.sqrt(((x-xj)*(x-xj))+((y-yj)*(y-yj))+((z-zj)*(z-zj)))
        g = ((1/(p*r))*(np.exp(-p*r)-1))+1
    
    return g

def calc_g_column2(x,y,z,xj,yj,zj,t=None):
    '''
    Returns the coefficients for spline interpolation using Green’s functions.
    '''
    assert isinstance(1.*x, np.ndarray), 'x must be a list' 
    assert isinstance(1.*y, np.ndarray), 'y must be a list' 
    assert isinstance(1.*z, np.ndarray), 'z must be a list' 
    assert (x.size == y.size == z.size), 'Sizes of x, y and z must be equal '   
    assert isinstance(1.*xj, float), 'xj must be a float' 
    assert isinstance(1.*yj, float), 'yj must be a float'
    assert isinstance(1.*zj, float), 'zj must be a float'
    
    if t is None:
        g = np.sqrt(((x-xj)*(x-xj))+((y-yj)*(y-yj))+((z-zj)*(z-zj)))
        
    else:
        assert (t>=0) & (t<1), 't must lie between 0 and 1'
        p = np.sqrt(t/(1-t))
        r = np.sqrt(((x-xj)*(x-xj))+((y-yj)*(y-yj))+((z-zj)*(z-zj)))
        pr = p*r
        g = (1/pr)*special.erf(pr/2)-(1/np.sqrt(np.pi))
    
    return g

def calc_A_by_column(x,y,z,xc,yc,zc,gcalc,t=None):
    '''
    Returns the sensibility matrix for spline interpolation.
    
    input
    x: np array - coordinates x of the data.
    y: np array - coordinates y of the data.
    z: np array - coordinates z of the data.
    
    xc: np array - coordinates x of the control points to calculate green function for spline interpolation.
    yc: np array - coordinates y of the control points to calculate green function for spline interpolation.
    zc: np array - coordinates z of the control points to calculate green function for spline interpolation.
    
    output
    A: np array 
    '''
    assert (x.size == y.size == z.size), 'Sizes of x, y and z must be equal '
    assert (xc.size == yc.size == zc.size), 'Sizes of xc, yc and zc must be equal '
    
    N = x.size
    M = xc.size

    A = np.zeros((N,M))
    
    if (gcalc == 1):
        for j,(xj,yj,zj) in enumerate(zip(xc,yc,zc)):
            A[:,j] = calc_g_column1(x,y,z,xj,yj,zj,t)
    else:
        for j,(xj,yj,zj) in enumerate(zip(xc,yc,zc)):
            A[:,j] = calc_g_column2(x,y,z,xj,yj,zj,t)
            
    return A
