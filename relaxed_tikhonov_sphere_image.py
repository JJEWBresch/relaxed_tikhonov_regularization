import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
from scipy.interpolate import CubicSpline
from scipy.stats import vonmises
from scipy.stats import vonmises_fisher
from scipy import interpolate
import matplotlib.gridspec
import condat_tv
from matplotlib import cm
from matplotlib.colors import LightSource

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def reconstruction(x):

    # angeled signal x given

    N = np.size(x)
    for j in range(3):
        for i in range(1,N):
            if np.abs(np.abs(x[i] - x[i-1])) > 0.9*np.pi:
                if x[i] - x[i-1] < 0:
                    x[i] = x[i] + 2*np.pi
                else:
                    x[i] = x[i] - 2*np.pi

    return x

def angle_S2(a, b, c):

    v = np.array([a, b, c]/np.sqrt(a**2+b**2+c**2))
    theta = np.arccos(v[2])
    phi = np.arctan2(b, a)

    return [phi, theta]

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def sample_vMF_image(N,d,kappa):

    dis = 0.1
    kap = 1

    x = np.linspace(1,N,N+1)
    x = np.arange(1,N+2,1)

    Vdata = np.random.randn(d,N+1,N+1)*kap

    x2 = np.arange(1,N,dis)

    l = np.size(x2)

    q = np.zeros((d,l,l))

    for i in range(d):
        fx = interpolate.interp2d(x, x, Vdata[i,:,:], kind='cubic')
        xx = fx(x2, x2)
        if i == d-1:
            scal = 1 
        else:
            scal = 0
        q[i,:,:] = scal + xx
    
    qq = np.zeros((d,l,l))

    for i in range(l):
        for j in range(l):
                q[:,i,j] = q[:,i,j]/np.linalg.norm(q[:,i,j])

                qq[:,i,j] = vonmises_fisher(mu=q[:,i,j], kappa=kappa).rvs(1)
                #kappa = 10
    
    return qq, q

def sample_vMF_noise_image(signal, kap):

    d, l, l = np.shape(signal)
    
    qq = np.zeros((d,l,l))

    for i in range(l):
        for j in range(l):
                qq[:,i,j] = vonmises_fisher(mu=signal[:,i,j], kappa=kap).rvs(1)
    
    return qq

def sample_toy_data(n, kap):

    x = np.zeros((2,n,n))

    x[0,0:int(n/3),0:int(n/3)] = 1
    x[1,0:int(n/3),0:int(n/3)] = 0

    x[0,int(n/3):n,0:int(n/3)] = 0
    x[1,int(n/3):n,0:int(n/3)] = 1

    x[0,0:int(n/2),int(n/3):n] = 0
    x[1,0:int(n/2),int(n/3):n] = -1

    x[0,int(n/2):n,int(n/3):n] = -1
    x[1,int(n/2):n,int(n/3):n] = 0

    y = sample_vMF_noise_image(x, kap)

    return [x,y]

def sample_toy_data_diag(n, kap):

    x = np.zeros((2,n,n))
    y = np.zeros((2,n,n))

    for i in range(n):
        for j in range(i):
            if j < 30+n-i:
                x[0,i,j] = 1
                x[1,i,j] = 0
            else:
                x[0,i,j] = 0
                x[1,i,j] = 1
        for j in range(i,n):
            if j < n-i:
                x[0,i,j] = -1
                x[1,i,j] = 0
            else:
                x[0,i,j] = 0
                x[1,i,j] = -1

    y = sample_vMF_noise_image(x, kap)

    return [x,y]

def angle_SO3(Data):

    for i in range(np.size(Data[0,0,:])):
        for j in range(np.size(Data[0,:,0])):
            if Data[0,i,j]<0:
                Data[:,i,j] = -Data[:,i,j]

    alpha = np.arctan2(np.sqrt(Data[1,:,:]**2+Data[2,:,:]**2+Data[3,:,:]**2), Data[0,:,:])
    v = np.array([Data[1,:,:], Data[2,:,:], Data[3,:,:]]/np.sqrt(Data[1,:,:]**2+Data[2,:,:]**2+Data[3,:,:]**2))
    theta = np.arccos(v[2])
    phi = np.arctan2(Data[2,:,:], Data[1,:,:])

    return [alpha, phi, theta]

def transformation_SO3(q):

    d, n, m = np.shape(q)
    qq = np.zeros((d,n,m))

    r1 = []
    r2 = []
    

    for i in range(n-1):
        for j in range(m-1):
            if np.arccos(np.sum(q[:,i,j]*q[:,i+1,j]))**2 > 3.9:
                q[:,i+1,j] = -q[:,i+1,j]
                r1 = np.append(r1, i+1)
                r2 = np.append(r2, j)
            if np.arccos(np.sum(q[:,i,j]*q[:,i,j+1]))**2 > 3.9:
                q[:,i,j+1] = -q[:,i,j+1]
                r1 = np.append(r1, i)
                r2 = np.append(r2, j+1)

    return [q, r1, r2]

def back_transformation_SO3(q, r1, r2):

    for i in range(np.size(r1)):
        q[:,int(r1[i]), int(r2[i])] = -q[:,int(r1[i]), int(r2[i])]
    
    return q

def sample_SO3_image(n):

    N = 10 
    dis = 0.1
    kap1 = 1
    kap2 = 1
    sig = 0.5
    d = 3

    x = np.linspace(1,N,N+1)
    x = np.arange(1,N+2,1)

    Vdata = np.random.randn(d,N+1,N+1)*kap1

    x2 = np.arange(1,N,dis)

    l = np.size(x2)

    q = np.zeros((d,l,l))

    for i in range(d):
        fx = interpolate.interp2d(x, x, Vdata[i,:,:], kind='cubic')
        xx = fx(x2, x2)
        if i == d-1:
            scal = 1 
        else:
            scal = 0
        q[i,:,:] = scal + xx
    
    qq = np.zeros((d,l,l))

    for i in range(l):
        for j in range(l):
                q[:,i,j] = q[:,i,j]/np.linalg.norm(q[:,i,j])

                qq[:,i,j] = vonmises_fisher(mu=q[:,i,j], kappa=30).rvs(1)
    
    NNoise , DData = sample_vMF_image(10,2, 15)

    AAx = np.angle(DData[0,:,:] + 1j*DData[1,:,:])
    ANx = np.angle(NNoise[0,:,:] + 1j*NNoise[1,:,:])

    nqq = np.zeros((d+1,l,l))
    sq = np.zeros((d+1,l,l))

    sq[0,:,:] = np.cos(AAx/2)
    sq[1:d+1,:,:] = q*np.sin(AAx/2)
    nqq[0,:,:] = np.cos(ANx/2)
    nqq[1:d+1,:,:] = qq*np.sin(ANx/2)
    
    return nqq, sq


def sample_torus_vMF_image(d, n, kap):

    dis = 0.1

    xxnew = np.arange(1, n, dis)

    Data = np.zeros((2*d, np.size(xxnew), np.size(xxnew)))
    Noise = np.zeros((2*d, np.size(xxnew), np.size(xxnew)))

    for i in range(d):
        noise, data = sample_vMF_image(n, 2, kap)
        Data[2*i: 2*i+2, :] = data
        Noise[2*i: 2*i+2, :] = noise
    
    return [Noise, Data]

def sample_smooth_hyperbolic_image(d,n):

    dis = 0.1
    kap = 1

    x = np.linspace(1,n,n+1)
    x = np.arange(1,n+2,1)

    Vdata = np.random.randn(d,n+1,n+1)*kap

    x1 = np.arange(1,n,dis)
    x21, x22 = np.meshgrid(x1, x1, indexing='ij', sparse='True')

    l = np.size(x1)

    q = np.zeros((d,l,l))

    for i in range(d):
        fx = interpolate.RegularGridInterpolator((x, x), Vdata[i,:,:], method='cubic')
        xx = fx((x21, x22))
        if i == d-1:
            scal = 1 
        else:
            scal = 0
        q[i,:,:] = scal + xx
    
    data = np.zeros((d+1,l,l))
    data[0:d,:,:] = q
    data[d,:,:] = np.sqrt(1 + np.sum(data**2,0))

    print('minkowsky inner-prod test : ', np.linalg.norm(1 + np.sum(data[0:d,:]**2,0) - data[d,:]**2))

    return data


#############################################
#
# operators
#
#############################################

def mod2pi(x):
    d = np.size(x)

    y = np.mod(x, 2*np.pi)
    
    y = np.where(y > np.pi, y-2*np.pi, y)

    return y

def proxCPPA(l,lam,reg,x,y):

    d = np.size(x)

    if l == 1:
        v = np.zeros(d)
        
        v = np.where(np.abs(x - y) <= np.pi, 0, np.sign(x - y))

        xx = mod2pi((x+lam*y)/(1+lam) + lam/(1+lam)*2*np.pi*v)
    
    if l == 2:
        r = np.size(x)
        y = x

        z = np.arange(r)
        z = np.where(np.mod(z,2) == 0, -1, 1)

        sf = [mod2pi(np.sum(y[2*i:2*i+2]*z[2*i:2*i+2])) for i in range(int(r/2))]

        s = np.sign(sf)

        m = np.where(np.abs(sf) < np.pi, np.minimum(lam*reg, np.abs(sf)/4), -np.minimum(lam*reg, np.pi/4))

        sm = s*m

        sm = [sm[int(i/2)] for i in range(r)]

        xx = mod2pi(y - sm*z)

    return xx

def L(x,r1,r2,N):

    U1 = np.zeros((N-1, N, 3,3), dtype='cdouble')

    U1[:,:,1,0] = x[0:N-1,:]
    U1[:,:,0,1] = np.conj(x[0:N-1,:])
    U1[:,:,2,0] = x[1:N,:]
    U1[:,:,0,2] = np.conj(x[1:N,:])
    U1[:,:,1,2] = r1
    U1[:,:,2,1] = np.conj(r1)

    U2 = np.zeros((N, N-1, 3,3), dtype='cdouble')

    U2[:,:,1,0] = x[:,0:N-1]
    U2[:,:,0,1] = np.conj(x[:,0:N-1])
    U2[:,:,2,0] = x[:,1:N]
    U2[:,:,0,2] = np.conj(x[:,1:N])
    U2[:,:,1,2] = r2
    U2[:,:,2,1] = np.conj(r2)

    return [U1, U2]

def adjL(U1, U2, N):

    x = np.zeros((N, N), dtype='cdouble')

    x[0:N-1,:] = U1[:,:,1,0] + np.matrix.conj(U1[:,:,0,1])
    x[1:N,:] = x[1:N,:] + np.squeeze(U1[:,:,2,0] + np.matrix.conj(U1[:,:,0,2]))
    r1 = np.squeeze(U1[:,:,1,2] + np.matrix.conj(U1[:,:,2,1]))

    x[:,0:N-1] = x[:,0:N-1] + U2[:,:,1,0] + np.matrix.conj(U2[:,:,0,1])
    x[:,1:N] = x[:,1:N] + np.squeeze(U2[:,:,2,0] + np.matrix.conj(U2[:,:,0,2]))
    r2 = np.squeeze(U2[:,:,1,2] + np.matrix.conj(U2[:,:,2,1]))

    return [x, r1, r2]

def L_red(x,r1,r2,N,d):

    U1 = np.zeros((N-1,N,d+2,d+2), dtype='float64')
    U2 = np.zeros((N,N-1,d+2,d+2), dtype='float64')

    U1[:,:,d,0:d] = np.transpose(x[:,0:N-1,:],(1,2,0))
    U1[:,:,d+1,0:d] = np.transpose(x[:,1:N,:],(1,2,0))

    U1[:,:,0:d,d] = np.transpose(x[:,0:N-1,:],(1,2,0))
    U1[:,:,0:d,d+1] = np.transpose(x[:,1:N,:],(1,2,0))

    U1[:,:,d,d+1] = r1
    U1[:,:,d+1,d] = r1

    U2[:,:,d,0:d] = np.transpose(x[:,:,0:N-1],(1,2,0))
    U2[:,:,d+1,0:d] = np.transpose(x[:,:,1:N],(1,2,0))

    U2[:,:,0:d,d] = np.transpose(x[:,:,0:N-1],(1,2,0))
    U2[:,:,0:d,d+1] = np.transpose(x[:,:,1:N],(1,2,0))

    U2[:,:,d,d+1] = r2
    U2[:,:,d+1,d] = r2

    return [U1, U2]

def adjL_red(U1,U2,N,d):

    x = np.zeros((d,N,N), dtype='float64')

    x[:,0:N-1,:] = 2*np.transpose(U1[:,:,0:d,d],(2,0,1))

    x[:,1:N,:] = x[:,1:N,:] + 2*np.transpose(U1[:,:,0:d,d+1],(2,0,1))

    r1 = 2*U1[:,:,d,d+1]

    x[:,:,0:N-1] = x[:,:,0:N-1] + 2*np.transpose(U2[:,:,0:d,d],(2,0,1))

    x[:,:,1:N] = x[:,:,1:N] + 2*np.transpose(U2[:,:,0:d,d+1],(2,0,1))

    r2 = 2*U2[:,:,d,d+1]

    return [x,r1,r2]

def dim_opL(a, b, c, d1, d2, e1, e2, f1, f2, g1, g2, N):

    # second dimension 

    U2 = np.zeros((N,N-1,6,6), dtype=complex)

    U2[:,:,0,2] = -1j*c[:,0:N-1]
    U2[:,:,0,3] = -b[:,0:N-1] - 1j*a[:,0:N-1]

    U2[:,:,2,0] = 1j*c[:,0:N-1]
    U2[:,:,3,0] = -b[:,0:N-1] + 1j*a[:,0:N-1]

    U2[:,:,1,2] = b[:,0:N-1] - 1j*a[:,0:N-1]
    U2[:,:,1,3] = 1j*c[:,0:N-1]

    U2[:,:,2,1] = b[:,0:N-1] + 1j*a[:,0:N-1]
    U2[:,:,3,1] = -1j*c[:,0:N-1]

    ###

    U2[:,:,0,4] = -1j*c[:,1:N]
    U2[:,:,0,5] = -b[:,1:N] - 1j*a[:,1:N]

    U2[:,:,4,0] = 1j*c[:,1:N]
    U2[:,:,5,0] = -b[:,1:N] + 1j*a[:,1:N]

    U2[:,:,1,4] = b[:,1:N] - 1j*a[:,1:N]
    U2[:,:,1,5] = 1j*c[:,1:N]

    U2[:,:,4,1] = b[:,1:N] + 1j*a[:,1:N]
    U2[:,:,5,1] = -1j*c[:,1:N]

    ###

    U2[:,:,2,4] = d2 - 1j*g2
    U2[:,:,2,5] = -f2 - 1j*e2
    
    U2[:,:,4,2] = d2 + 1j*g2
    U2[:,:,5,2] = -f2 + 1j*e2

    U2[:,:,3,4] = f2 - 1j*e2
    U2[:,:,3,5] = d2 + 1j*g2

    U2[:,:,4,3] = f2 + 1j*e2
    U2[:,:,5,3] = d2 - 1j*g2

    # first dimension 

    U1 = np.zeros((N-1,N,6,6), dtype=complex)

    U1[:,:,0,2] = -1j*c[0:N-1,:]
    U1[:,:,0,3] = -b[0:N-1,:] - 1j*a[0:N-1,:]

    U1[:,:,2,0] = 1j*c[0:N-1,:]
    U1[:,:,3,0] = -b[0:N-1,:] + 1j*a[0:N-1,:]

    U1[:,:,1,2] = b[0:N-1,:] - 1j*a[0:N-1,:]
    U1[:,:,1,3] = 1j*c[0:N-1,:]

    U1[:,:,2,1] = b[0:N-1,:] + 1j*a[0:N-1,:]
    U1[:,:,3,1] = -1j*c[0:N-1,:]

    ###

    U1[:,:,0,4] = -1j*c[1:N,:]
    U1[:,:,0,5] = -b[1:N,:] - 1j*a[1:N,:]

    U1[:,:,4,0] = 1j*c[1:N,:]
    U1[:,:,5,0] = -b[1:N,:] + 1j*a[1:N,:]

    U1[:,:,1,4] = b[1:N,:] - 1j*a[1:N,:]
    U1[:,:,1,5] = 1j*c[1:N,:]

    U1[:,:,4,1] = b[1:N,:] + 1j*a[1:N,:]
    U1[:,:,5,1] = -1j*c[1:N,:]

    ###

    U1[:,:,2,4] = d1 - 1j*g1
    U1[:,:,2,5] = -f1 - 1j*e1
    
    U1[:,:,4,2] = d1 + 1j*g1
    U1[:,:,5,2] = -f1 + 1j*e1

    U1[:,:,3,4] = f1 - 1j*e1
    U1[:,:,3,5] = d1 + 1j*g1

    U1[:,:,4,3] = f1 + 1j*e1
    U1[:,:,5,3] = d1 - 1j*g1

    return [U1, U2]

def dim_adjL(U1, U2, N):
        
    x = np.zeros((N,N), dtype=complex)
    y = np.zeros((N,N), dtype=complex)
    z = np.zeros((N,N), dtype=complex)

    x[0:N-1,:] = np.imag(U1[:,:,2,1] - U1[:,:,0,3] - np.matrix.conj(U1[:,:,3,0]) + np.matrix.conj(U1[:,:,1,2]))
    x[1:N,:] = x[1:N,:] + np.squeeze(np.imag(U1[:,:,4,1] - U1[:,:,0,5] - np.matrix.conj(U1[:,:,5,0]) + np.matrix.conj(U1[:,:,1,4])))
    
    x[:,0:N-1] = x[:,0:N-1] + np.imag(U2[:,:,2,1] - U2[:,:,0,3] - np.matrix.conj(U2[:,:,3,0]) + np.matrix.conj(U2[:,:,1,2]))
    x[:,1:N] = x[:,1:N] + np.squeeze(np.imag(U2[:,:,4,1] - U2[:,:,0,5] - np.matrix.conj(U2[:,:,5,0]) + np.matrix.conj(U2[:,:,1,4])))
    
    d1 = np.squeeze(np.real(np.matrix.conj(U1[:,:,2,4]) + np.matrix.conj(U1[:,:,5,3]) + U1[:,:,3,5] + U1[:,:,4,2]))
    e1 = -np.squeeze(np.imag(np.matrix.conj(U1[:,:,5,2]) - np.matrix.conj(U1[:,:,3,4]) + U1[:,:,2,5] - U1[:,:,4,3]))
    f1 = -np.squeeze(np.real(np.matrix.conj(U1[:,:,5,2]) - np.matrix.conj(U1[:,:,3,4]) + U1[:,:,2,5] - U1[:,:,4,3]))
    g1 = np.squeeze(np.imag(np.matrix.conj(U1[:,:,2,4]) + np.matrix.conj(U1[:,:,5,3]) + U1[:,:,3,5] + U1[:,:,4,2]))


    y[:,0:N-1] = np.real(U2[:,:,2,1] - U2[:,:,0,3] - np.matrix.conj(U2[:,:,3,0]) + np.matrix.conj(U2[:,:,1,2]))
    y[:,1:N] = y[:,1:N] + np.squeeze(np.real(U2[:,:,4,1] - U2[:,:,0,5] - np.matrix.conj(U2[:,:,5,0]) + np.matrix.conj(U2[:,:,1,4])))
    
    y[0:N-1,:] = y[0:N-1,:] + np.real(U1[:,:,2,1] - U1[:,:,0,3] - np.matrix.conj(U1[:,:,3,0]) + np.matrix.conj(U1[:,:,1,2]))
    y[1:N,:] = y[1:N,:] + np.squeeze(np.real(U1[:,:,4,1] - U1[:,:,0,5] - np.matrix.conj(U1[:,:,5,0]) + np.matrix.conj(U1[:,:,1,4])))

    d2 = np.squeeze(np.real(np.matrix.conj(U2[:,:,2,4]) + np.matrix.conj(U2[:,:,5,3]) + U2[:,:,3,5] + U2[:,:,4,2]))
    e2 = -np.squeeze(np.imag(np.matrix.conj(U2[:,:,5,2]) - np.matrix.conj(U2[:,:,3,4]) + U2[:,:,2,5] - U2[:,:,4,3]))
    f2 = -np.squeeze(np.real(np.matrix.conj(U2[:,:,5,2]) - np.matrix.conj(U2[:,:,3,4]) + U2[:,:,2,5] - U2[:,:,4,3]))
    g2 = np.squeeze(np.imag(np.matrix.conj(U2[:,:,2,4]) + np.matrix.conj(U2[:,:,5,3]) + U2[:,:,3,5] + U2[:,:,4,2]))
 
    z[:,0:N-1] = z[:,0:N-1] + np.imag(U2[:,:,2,0] - U2[:,:,0,2] + np.matrix.conj(U2[:,:,3,1]) - np.matrix.conj(U2[:,:,1,3]))
    z[:,1:N] = z[:,1:N] + np.squeeze(np.imag(U2[:,:,4,0] - U2[:,:,0,4] + np.matrix.conj(U2[:,:,5,1]) - np.matrix.conj(U2[:,:,1,5])))

    z[0:N-1,:] = z[0:N-1,:] + np.imag(U1[:,:,2,0] - U1[:,:,0,2] + np.matrix.conj(U1[:,:,3,1]) - np.matrix.conj(U1[:,:,1,3]))
    z[1:N,:] = z[1:N,:] + np.squeeze(np.imag(U1[:,:,4,0] - U1[:,:,0,4] + np.matrix.conj(U1[:,:,5,1]) - np.matrix.conj(U1[:,:,1,5])))
    
    return [x, y, z, d1, d2, e1, e2, f1, f2, g1, g2]

def opL(w, x, y, z, l1, l2, f1, f2, g1, g2, h1, h2, N):

    U1 = np.zeros((N-1,N,6,6), dtype=complex)

    U1[:,:,0,2] = w[0:N-1,:] - 1j*x[0:N-1,:]
    U1[:,:,0,3] = -y[0:N-1,:] - 1j*z[0:N-1,:]

    U1[:,:,2,0] = w[0:N-1,:] + 1j*x[0:N-1,:]
    U1[:,:,3,0] = -y[0:N-1,:] + 1j*z[0:N-1,:]

    U1[:,:,1,2] = y[0:N-1,:] - 1j*z[0:N-1,:]
    U1[:,:,1,3] = w[0:N-1,:] + 1j*x[0:N-1,:]

    U1[:,:,2,1] = y[0:N-1,:] + 1j*z[0:N-1,:]
    U1[:,:,3,1] = w[0:N-1,:] - 1j*x[0:N-1,:]

    ###

    U1[:,:,0,4] = w[1:N,:] - 1j*x[1:N,:]
    U1[:,:,0,5] = -y[1:N,:] - 1j*z[1:N,:]

    U1[:,:,4,0] = w[1:N,:] + 1j*x[1:N,:]
    U1[:,:,5,0] = -y[1:N,:] + 1j*z[1:N,:]

    U1[:,:,1,4] = y[1:N,:] - 1j*z[1:N,:]
    U1[:,:,1,5] = w[1:N,:] + 1j*x[1:N,:]

    U1[:,:,4,1] = y[1:N,:] + 1j*z[1:N,:]
    U1[:,:,5,1] = w[1:N,:] - 1j*x[1:N,:]

    ###

    U1[:,:,2,4] = l1 - 1j*f1
    U1[:,:,2,5] = -g1 - 1j*h1
    
    U1[:,:,4,2] = l1 + 1j*f1
    U1[:,:,5,2] = -g1 + 1j*h1

    U1[:,:,3,4] = g1 - 1j*h1
    U1[:,:,3,5] = l1 + 1j*f1

    U1[:,:,4,3] = g1 + 1j*h1
    U1[:,:,5,3] = l1 - 1j*f1

    # second dimension

    U2 = np.zeros((N,N-1,6,6), dtype=complex)

    U2[:,:,0,2] = w[:,0:N-1] - 1j*x[:,0:N-1]
    U2[:,:,0,3] = -y[:,0:N-1] - 1j*z[:,0:N-1]

    U2[:,:,2,0] = w[:,0:N-1] + 1j*x[:,0:N-1]
    U2[:,:,3,0] = -y[:,0:N-1] + 1j*z[:,0:N-1]

    U2[:,:,1,2] = y[:,0:N-1] - 1j*z[:,0:N-1]
    U2[:,:,1,3] = w[:,0:N-1] + 1j*x[:,0:N-1]

    U2[:,:,2,1] = y[:,0:N-1] + 1j*z[:,0:N-1]
    U2[:,:,3,1] = w[:,0:N-1] - 1j*x[:,0:N-1]

    ###

    U2[:,:,0,4] = w[:,1:N] - 1j*x[:,1:N]
    U2[:,:,0,5] = -y[:,1:N] - 1j*z[:,1:N]

    U2[:,:,4,0] = w[:,1:N] + 1j*x[:,1:N]
    U2[:,:,5,0] = -y[:,1:N] + 1j*z[:,1:N]

    U2[:,:,1,4] = y[:,1:N] - 1j*z[:,1:N]
    U2[:,:,1,5] = w[:,1:N] + 1j*x[:,1:N]

    U2[:,:,4,1] = y[:,1:N] + 1j*z[:,1:N]
    U2[:,:,5,1] = w[:,1:N] - 1j*x[:,1:N]

    ###

    U2[:,:,2,4] = l2 - 1j*f2
    U2[:,:,2,5] = -g2 - 1j*h2
    
    U2[:,:,4,2] = l2 + 1j*f2
    U2[:,:,5,2] = -g2 + 1j*h2

    U2[:,:,3,4] = g2 - 1j*h2
    U2[:,:,3,5] = l2 + 1j*f2

    U2[:,:,4,3] = g2 + 1j*h2
    U2[:,:,5,3] = l2 - 1j*f2

    return [U1, U2]

def adjopL(U1, U2, N):
    
    w = np.zeros((N,N))
    x = np.zeros((N,N))
    y = np.zeros((N,N))
    z = np.zeros((N,N))
    
    w[0:N-1,:] = np.real(U1[:,:,2,0] + U1[:,:,3,1] + U1[:,:,0,2] + U1[:,:,1,3])
    w[1:N,:] = w[1:N,:] + np.real(U1[:,:,4,0] + U1[:,:,5,1] + U1[:,:,0,4] + U1[:,:,1,5])

    w[:,0:N-1] = w[:,0:N-1] + np.real(U2[:,:,2,0] + U2[:,:,3,1] + U2[:,:,0,2] + U2[:,:,1,3])
    w[:,1:N] = w[:,1:N] + np.real(U2[:,:,4,0] + U2[:,:,5,1] + U2[:,:,0,4] + U2[:,:,1,5])

    x[0:N-1,:] = np.imag(U1[:,:,2,0] + np.matrix.conj(U1[:,:,3,1]) + np.matrix.conj(U1[:,:,0,2]) + U1[:,:,1,3])
    x[1:N,:] = x[1:N,:] + np.imag(U1[:,:,4,0] + np.matrix.conj(U1[:,:,5,1]) + np.matrix.conj(U1[:,:,0,4]) + U1[:,:,1,5])

    x[:,0:N-1] = x[:,0:N-1] + np.imag(U2[:,:,2,0] + np.matrix.conj(U2[:,:,3,1]) + np.matrix.conj(U2[:,:,0,2]) + U2[:,:,1,3])
    x[:,1:N] = x[:,1:N] + np.imag(U2[:,:,4,0] + np.matrix.conj(U2[:,:,5,1]) + np.matrix.conj(U2[:,:,0,4]) + U2[:,:,1,5])

    y[0:N-1,:] = np.real(U1[:,:,2,1] - U1[:,:,3,0] - U1[:,:,0,3] + U1[:,:,1,2])
    y[1:N,:] = y[1:N,:] + np.real(U1[:,:,4,1] - U1[:,:,5,0] - U1[:,:,0,5] + U1[:,:,1,4])

    y[:,0:N-1] = y[:,0:N-1] + np.real(U2[:,:,2,1] - U2[:,:,3,0] - U2[:,:,0,3] + U2[:,:,1,2])
    y[:,1:N] = y[:,1:N] + np.real(U2[:,:,4,1] - U2[:,:,5,0] - U2[:,:,0,5] + U2[:,:,1,4])

    z[0:N-1,:] = np.imag(U1[:,:,2,1] + U1[:,:,3,0] + np.matrix.conj(U1[:,:,0,3]) + np.matrix.conj(U1[:,:,1,2]))
    z[1:N,:] = z[1:N,:] + np.imag(U1[:,:,4,1] + U1[:,:,5,0] + np.matrix.conj(U1[:,:,0,5]) + np.matrix.conj(U1[:,:,1,4]))

    z[:,0:N-1] = z[:,0:N-1] + np.imag(U2[:,:,2,1] + U2[:,:,3,0] + np.matrix.conj(U2[:,:,0,3]) + np.matrix.conj(U2[:,:,1,2]))
    z[:,1:N] = z[:,1:N] + np.imag(U2[:,:,4,1] + U2[:,:,5,0] + np.matrix.conj(U2[:,:,0,5]) + np.matrix.conj(U2[:,:,1,4]))
    
    ''' where the complex conjugation, has to be shifted onto the operator L^* since 
            < Lq , U >_F = < q , L^*U >_R
        which is just definied be mulitplication with, since we are not on C^N but R^(4)^N includes q
    '''

    l1 = np.real(U1[:,:,4,2] + U1[:,:,5,3] + U1[:,:,2,4] + U1[:,:,3,5])
    f1 = np.imag(U1[:,:,4,2] + np.conjugate(U1[:,:,5,3]) + np.conjugate(U1[:,:,2,4]) + U1[:,:,3,5])
    g1 = np.real(U1[:,:,4,3] - U1[:,:,5,2] - U1[:,:,2,5] + U1[:,:,3,4])
    h1 = np.imag(U1[:,:,4,3] + U1[:,:,5,2] + np.conjugate(U1[:,:,2,5]) + np.conjugate(U1[:,:,3,4]))

    l2 = np.real(U2[:,:,4,2] + U2[:,:,5,3] + U2[:,:,2,4] + U2[:,:,3,5])
    f2 = np.imag(U2[:,:,4,2] + np.conjugate(U2[:,:,5,3]) + np.conjugate(U2[:,:,2,4]) + U2[:,:,3,5])
    g2 = np.real(U2[:,:,4,3] - U2[:,:,5,2] - U2[:,:,2,5] + U2[:,:,3,4])
    h2 = np.imag(U2[:,:,4,3] + U2[:,:,5,2] + np.conjugate(U2[:,:,2,5]) + np.conjugate(U2[:,:,3,4]))
    
    return [w, x, y, z, l1, l2, f1, f2, g1, g2, h1, h2]


def prox(U1, U2, N):

    for i in range(0,N-1):
        for j in range(0,N):
            [D, V] = np.linalg.eigh(U1[i,j,:,:])
            D = np.diag(np.minimum(np.real(D),0))
            U1[i,j,:,:] = V@D@np.transpose(np.conjugate(V))
    
    for i in range(0,N):
        for j in range(0,N-1):
            [D, V] = np.linalg.eigh(U2[i,j,:,:])
            D = np.diag(np.minimum(np.real(D),0))
            U2[i,j,:,:] = V@D@np.transpose(np.conjugate(V))

    return [U1, U2]

def dim_prox(U1, U2, N):

    for i in range(0,N-1):
        for j in range(0,N):
                [D, V] = np.linalg.eigh(U1[i,j,:,:])
                D = np.diag(np.minimum(np.real(D),0))
                U1[i,j,:,:] = V@D@np.transpose(np.conjugate(V))
    
    for i in range(0,N):
        for j in range(0,N-1):
                [D, V] = np.linalg.eigh(U2[i,j,:,:])
                D = np.diag(np.minimum(np.real(D),0))
                U2[i,j,:,:] = V@D@np.transpose(np.conjugate(V))

    return [U1, U2]

def dim_proxADMM(U1, U2, N):

    for i in range(0,N-1):
        for j in range(0,N):
                [D, V] = np.linalg.eigh(U1[i,j,:,:])
                D = np.diag(np.maximum(np.real(D),-1))
                U1[i,j,:,:] = V@D@np.transpose(np.conjugate(V))
    
    for i in range(0,N):
        for j in range(0,N-1):
                [D, V] = np.linalg.eigh(U2[i,j,:,:])
                D = np.diag(np.maximum(np.real(D),-1))
                U2[i,j,:,:] = V@D@np.transpose(np.conjugate(V))

    return [U1, U2]

def grad(x):
    
    d = np.diff(x,1,0)
    d2 = np.diff(x,1,1)

    N = np.size(x[:,0])

    q = np.zeros((N,N))
    q2 = np.zeros((N,N))

    q[0,:] = -d[0,:]
    q[1:N-1,:] = -np.diff(d,1,0)
    q[N-1,:] = d[N-2,:]

    q2[:,0] = -d2[:,0]
    q2[:,1:N-1] = -np.diff(d2,1,1)
    q[:,N-1] = d2[:,N-2]

    return q+q2

def proxADMM(U1, U2, N):

    for i in range(0,N-1):
        for j in range(0,N):
            [D, V] = np.linalg.eigh(U1[i,j,:,:])
            D = np.diag(np.maximum(np.real(D),-1))
            U1[i,j,:,:] = V@D@np.transpose(np.conjugate(V))
    
    for k in range(0,N):
        for l in range(0,N-1):
            [D, V] = np.linalg.eigh(U2[k,l,:,:])
            D = np.diag(np.maximum(np.real(D),-1))
            U2[k,l,:,:] = V@D@np.transpose(np.conjugate(V))

    return [U1, U2]

def proj_B1(x):

        r, d, d = np.shape(x)

        norm = np.sqrt(np.sum(x**2,0))
        
        for i in range(d):
            for j in range(d):
                #if np.linalg.norm(x[i]) > 1:
                #    x[i] = x[i]/np.linalg.norm(x[i])
                if norm[i,j] > 1:
                    x[:,i,j] = x[:,i,j]/norm[i,j]
            
        return x

def opL_2DHyper(x, v, f1, ll1, f2, ll2):

    d, N, N = np.shape(x)
    l = d-1

    L2 = np.zeros((d+4, d+4, N, N-1))

    L2[l+1,0:d,:,:] = x[:,:,0:N-1]
    L2[l+2,0:d,:,:] = x[:,:,0:N-1]
    L2[l+2,d-1,:,:] = -L2[l+2,d-1,:,:]
    L2[l+3,0:d,:,:] = x[:,:,1:N]
    L2[l+4,0:d,:,:] = x[:,:,1:N]
    L2[l+4,d-1,:,:] = -L2[l+4,d-1,:,:]

    L2[0:d,l+1,:,:] = x[:,:,0:N-1]
    L2[0:d,l+2,:,:] = x[:,:,0:N-1]
    L2[d-1,l+2,:,:] = -L2[d-1,l+2,:,:]
    L2[0:d,l+3,:,:] = x[:,:,1:N]
    L2[0:d,l+4,:,:] = x[:,:,1:N]
    L2[d-1,l+4,:,:] = -L2[d-1,l+4,:,:]

    ####

    L2[d, d, :,:] = v[:,0:N-1]
    L2[d+1, d+1,:, :] = v[:,0:N-1]
    L2[d+2, d+2, :,:] = v[:,1:N]
    L2[d+3, d+3, :,:] = v[:,1:N]

    ####

    L2[d+2, d,:,:] = f2
    L2[d+3, d+1,:,:] = f2
    L2[d, d+2,:,:] = f2
    L2[d+1, d+3,:,:] = f2

    ####

    L2[d+2, d+1,:,:] = ll2
    L2[d+3, d,:,:] = ll2
    L2[d+1, d+2,:,:] = ll2
    L2[d, d+3,:,:] = ll2

    ###
    ###
    ###

    L1 = np.zeros((d+4, d+4, N-1, N))

    L1[l+1,0:d,:,:] = x[:,0:N-1,:]
    L1[l+2,0:d,:,:] = x[:,0:N-1,:]
    L1[l+2,d-1,:,:] = -L1[l+2,d-1,:,:]
    L1[l+3,0:d,:,:] = x[:,1:N,:]
    L1[l+4,0:d,:,:] = x[:,1:N,:]
    L1[l+4,d-1,:,:] = -L1[l+4,d-1,:,:]

    L1[0:d,l+1,:,:] = x[:,0:N-1,:]
    L1[0:d,l+2,:,:] = x[:,0:N-1,:]
    L1[d-1,l+2,:,:] = -L1[d-1,l+2,:,:]
    L1[0:d,l+3,:,:] = x[:,1:N,:]
    L1[0:d,l+4,:,:] = x[:,1:N,:]
    L1[d-1,l+4,:,:] = -L1[d-1,l+4,:,:]

    ####

    L1[d, d, :,:] = v[0:N-1,:]
    L1[d+1, d+1, :,:] = v[0:N-1,:]
    L1[d+2, d+2, :,:] = v[1:N,:]
    L1[d+3, d+3, :,:] = v[1:N,:]

    ####

    L1[d+2, d,:,:] = f1
    L1[d+3, d+1,:,:] = f1
    L1[d, d+2,:,:] = f1
    L1[d+1, d+3,:,:] = f1

    ####

    L1[d+2, d+1,:,:] = ll1
    L1[d+3, d,:,:] = ll1
    L1[d+1, d+2,:,:] = ll1
    L1[d, d+3,:,:] = ll1

    return [L1, L2]

def adjopL_2DHyper(U1, U2):

    r,r,M,N = np.shape(U1)
    #print(np.shape(U))

    N = M + 1
    d = r-4
    #print(d, N)

    x = np.zeros((d, N, N))
    v = np.zeros((N,N))
    f1 = np.zeros((M,N))
    l1 = np.zeros((M,N))
    f2 = np.zeros((N,M))
    l2 = np.zeros((N,M))

    x[:,0:N-1,:] += U1[r-4,0:d,:] + U1[r-3,0:d,:] + U1[0:d,r-4,:] + U1[0:d,r-3,:]
    x[d-1,0:N-1,:] += U1[r-4,d-1,:] - U1[r-3,d-1,:] + U1[d-1,r-4,:] - U1[d-1,r-3,:]
    x[:,1:N,:] += U1[r-2,0:d,:] + U1[r-1,0:d,:] + U1[0:d,r-2,:] + U1[0:d,r-1,:]
    x[d-1,1:N,:] += U1[r-2,d-1,:] - U1[r-1,d-1,:] + U1[d-1,r-2,:] - U1[d-1,r-1,:]

    x[:,:,0:N-1] += U2[r-4,0:d,:,:] + U2[r-3,0:d,:,:] + U2[0:d,r-4,:,:] + U2[0:d,r-3,:,:]
    x[d-1,:,0:N-1] += U2[r-4,d-1,:,:] - U2[r-3,d-1,:,:] + U2[d-1,r-4,:,:] - U2[d-1,r-3,:,:]
    x[:,:,1:N] += U2[r-2,0:d,:,:] + U2[r-1,0:d,:,:] + U2[0:d,r-2,:,:] + U2[0:d,r-1,:,:]
    x[d-1,:,1:N] += U2[r-2,d-1,:,:] - U2[r-1,d-1,:,:] + U2[d-1,r-2,:,:] - U2[d-1,r-1,:,:]

    v[0:N-1,:] = U1[r-4,r-4,:,:] + U1[r-3,r-3,:,:]
    v[1:N,:] += U1[r-2,r-2,:,:] + U1[r-1,r-1,:,:]
    v[:,0:N-1] += U2[r-4,r-4,:,:] + U2[r-3,r-3,:,:]
    v[:,1:N] += U2[r-2,r-2,:,:] + U2[r-1,r-1,:,:]

    f1 = U1[r-2,r-4,:,:] + U1[r-1,r-3,:,:] + U1[r-4,r-2,:,:] + U1[r-3,r-1,:,:]
    l1 = U1[r-1,r-4,:,:] + U1[r-2,r-3,:,:] + U1[r-3,r-2,:,:] + U1[r-4,r-1,:,:]

    f2 = U2[r-2,r-4,:,:] + U2[r-1,r-3,:,:] + U2[r-4,r-2,:,:] + U2[r-3,r-1,:,:]
    l2 = U2[r-1,r-4,:,:] + U2[r-2,r-3,:,:] + U2[r-3,r-2,:,:] + U2[r-4,r-1,:,:]

    return [x, v, f1, l1, f2, l2]

def prox_2DHyper(W):

    r,r,M1,M2 = np.shape(W)

    WW = np.zeros((r,r,M1,M2))
    for i in range(M1):
        for j in range(M2):
            D, R = np.linalg.eig(W[:,:,i,j])
            DD = np.maximum(np.real(D), 0)
            WW[:,:,i,j] = R@np.diag(DD)@R.T
    return WW


#############################################
#
# solvers
#
#############################################

def CPPA2D(y, y0, reg, lam0):

    print('iter. \t| funv-value \t\t| RMSE \t\t| error')
    print('---------------------------------------------------------------')
    d, d = np.shape(y0)
    x = np.zeros((d,d))
    x1 = np.random.rand(d,d)
    k = 1
    while np.linalg.norm(x - x1) > 1e-4:
    #while k < 1000: 
            lam = lam0/k
            if np.mod(k,100) == 0:
                print(k, '\t|', "%10.3e"%(np.linalg.norm(np.cos(x) + 1j*np.sin(x) - np.cos(y) - 1j*np.sin(y))**2 + reg*np.sum(np.abs(np.cos(x[0:d-1,:] - x[1:d,:])) + np.abs(np.sin(x[0:d-1,:] - x[1:d,:]))) + reg*np.sum(np.abs(np.cos(x[:,0:d-1] - x[:,1:d])) + np.abs(np.sin(x[:,0:d-1] - x[:,1:d])))), '\t|', "%10.2e"% (np.linalg.norm(np.cos(x) + 1j*np.sin(x) - np.cos(y0) - 1j*np.sin(y0))**2/np.size(x)),' \t|', "%10.2e"% np.linalg.norm(x - x1))
            x1 = x

            #x = prox2D(1,lam,reg,x,y)
            x = proxCPPA(1,lam,reg,x,y)
            
            for j in range(d):
                x[0:2*int((d-1)/2)+2,j] = proxCPPA(2,lam,reg,x[0:2*int((d-1)/2)+2,j],y[0:2*int((d-1)/2)+2,j])
                x[1:2*int((d-1)/2)+1,j] = proxCPPA(2,lam,reg,x[1:2*int((d-1)/2)+1,j],y[1:2*int((d-1)/2)+1,j])
            for i in range(d):
                x[i,0:2*int((d-1)/2)+2] = proxCPPA(2,lam,reg,x[i,0:2*int((d-1)/2)+2],y[i,0:2*int((d-1)/2)+2])
                x[i,1:2*int((d-1)/2)+1] = proxCPPA(2,lam,reg,x[i,1:2*int((d-1)/2)+1],y[i,1:2*int((d-1)/2)+1])

            k = k+1
    
    datei = open('data_S1_2Dgrid_angle.txt','a')
    datei.write(str(np.linalg.norm(np.cos(x) + 1j*np.sin(x) - np.cos(y0) - 1j*np.sin(y0))**2))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(np.cos(x) - np.cos(y0)) + np.abs(np.sin(x) - np.sin(y0)))))
    datei.write('\n')
    datei.close()
    
    return x

def ADRA(y, gam, mu, iter):

    s = y
    x = y
    d, d = np.shape(y)
    r = np.zeros(np.shape(y))

    n = 0
    xx = np.random.rand(d,d)

    #for n in range(iter):
    while np.linalg.norm(x - xx) > 10-6:
        xx = x
        r1 = r
        r = s - x + condat_tv.tv_denoise_matrix(1/(1+gam)*(2*x - s) + gam/(1+gam)*y, mu*gam/(1+gam))
        s = r + n/(n+3)*(r - r1)
        x = condat_tv.tv_denoise_matrix(s.T , mu*gam)
        x = x.T

        n = n+1

    return x

def ADMM_TV_BOX(y, y0, mu, rho, iter):

        r, d, d = np.shape(y)
        f = r-1

        x = np.zeros((r, d, d))
        u = np.zeros((r, d, d))
        z = np.zeros((r, d, d))

        print('iter. \t| funv-value \t| RMSE \t\t| error \t| spherical-error')
        print('----------------------------------------------------------------------------------')

        x11 = np.random.randn(r, d, d)

        j = 0
        flag = 1

        #for i in range(iter):
        while np.linalg.norm(x11 - x) > 1e-6 and flag > 1e-4 and j < 30000:

            x11 = np.copy(x)

            # argmin_x  -<x,y> + mu|Dx|_1 + iota(u) + rho/2||x - u + z||_2
            for l in range(r):
                x[l,:,:] = ADRA(u[l,:,:] - z[l,:,:] + y[l,:,:]/rho, 1, mu/rho, 300)

            flag = np.linalg.norm(1 - np.sqrt(np.sum(x**2,0)))

            # proj_B(1)
            u = proj_B1(x + z)

            # update
            z = z + x - u
        
            if np.mod(j,100) == 0:
                res_tv = 0
                for l in range(r):
                    res_tv += mu*np.sum(np.abs(x[l,0:d-1,:] - x[l,1:d,:])) + mu*np.sum(np.abs(x[l,:,0:d-1] - x[l,:,1:d]))

                print(j, ' \t| ', "%10.3e"%(np.linalg.norm(x - y)**2 + res_tv) , ' \t| ', "%10.3e"% (np.linalg.norm(x - y0)**2/d**2), ' \t| ' , "%10.3e"% (np.linalg.norm(x11 - x)), ' \t| ' , "%10.3e"% (np.linalg.norm(1 - np.sqrt(np.sum(x**2,0)))))

            j = j+1

        ame = 0
        for l in range(r):
            ame += np.sum(np.abs(x[l,:,:] - y0[l,:,:]))
        
        datei = open('data_S{0}_2Dgrid_box.txt'.format(f),'a')
        datei.write(str(np.linalg.norm(x - y0)**2))
        datei.write('\n')
        datei.write(str(ame))
        datei.write('\n')
        datei.close()

        return [x, z]

def Condat_TV_proj(y, y0, lam, rho, eps, iter):

    y1 = y[0,:,:]
    y2 = y[1,:,:]

    x1 = ADRA(y1, 1, lam, 300)
    x2 = ADRA(y2, 1, lam, 300)

    x = x1 + 1j*x2

    xp = proj_B1(x)

    return xp

def PMM_S1_2D(y, y0, lam, iter, tau, circ, eps):

    N = np.size(y[:,0]) 

    #x = y
    x = np.zeros((N,N), dtype=complex)
    r1 = x[0:N-1,:]*np.conjugate(x[1:N,:])
    r2 = x[:,0:N-1]*np.conjugate(x[:,1:N])

    U1 = np.zeros((N-1, N, 3,3), dtype=complex)
    U2 = np.zeros((N, N-1, 3,3), dtype=complex)

    Id1 = np.zeros((N-1,N, 3,3), dtype=complex)
    Id2 = np.zeros((N,N-1, 3,3), dtype=complex)
    
    Id1[:,:,0,0] = 1 + 0*1j
    Id1[:,:,1,1] = 1 + 0*1j
    Id1[:,:,2,2] = 1 + 0*1j
    Id2[:,:,0,0] = 1 + 0*1j
    Id2[:,:,1,1] = 1 + 0*1j
    Id2[:,:,2,2] = 1 + 0*1j

    sigma = 1/(8*tau)

    k = 0

    data = np.zeros(iter)
    datatime = np.zeros(iter)
    datares1 = np.zeros(iter)
    datares2 = np.zeros(iter)
    flagggg = 0

    print('iteration \t| func-value \t| non-convex-cost \t| spherical-error')
    print('----------------------------------------------------------------------')

    starttime = time.time()

    #print(np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(x[0:N-1,:]*np.conjugate(x[1:N,:]))) + lam*np.sum(1 - np.real(x[:,0:N-1]*np.conjugate(x[:,1:N]))))
    #print(np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(r1)) + lam*np.sum(1 - np.real(r2)))

    for s in range(0,iter):
        [a_x, a_r1, a_r2] = adjL(U1, U2, N)
        a_x = a_x - y
        a_r1 = a_r1 - lam
        a_r2 = a_r2 - lam

        if circ == 0:
            [X1, X2] = L(x - 2*tau*a_x, r1 - 2*tau*a_r1, r2 - 2*tau*a_r2, N)

            [U1, U2] = prox(U1 + sigma*(X1 + Id1), U2 + sigma*(X2 + Id2), N)

            x = x - tau*a_x
            r1 = r1 - tau*a_r1
            r2 = r2 - tau*a_r2

        
        if circ == 1:
            [X1, X2] = L((x  - 2*tau*a_x)/np.abs(x  - 2*tau*a_x), r1 - 2*tau*a_r1, r2 - 2*tau*a_r2, N)

            [U1, U2] = prox(U1 + sigma*(X1 + Id1), U2 + sigma*(X2 + Id2), N)

            x = x - tau*a_x
            x = x/np.abs(x)

            r1 = r1 - tau*a_r1
            r2 = r2 - tau*a_r2

        data[s] = np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(r1)) + lam*np.sum(1 - np.real(r2))

        datares1[s] = np.linalg.norm(a_r1)+np.linalg.norm(a_r2)
        datares2[s] = np.linalg.norm(a_x)
        datatime[s] = time.time()
        if datares1[s] + datares2[s] < 10**(-eps) and flagggg == 0:
            print('iteration :', s, datatime[s] - starttime)
            flagggg = 1

        if np.mod(s,100)==0:

            k = k + 1
            #print(datares1[s], datares2[s])
            
            print(s, '\t\t| ', "%10.3e"%(data[s]), '\t| ',  "%10.3e"%(np.sum(1-np.cos(np.angle(x)-np.angle(y0))) + lam*np.sum(1-np.cos(np.angle(x[0:N-1,:])-np.angle(x[1:N,:]))) + lam*np.sum(1-np.cos(np.angle(x[:,0:N-1])-np.angle(x[:,1:N])))), '\t\t| ', "%10.3e"%(1 - np.mean(np.abs(x))))
    
    flag = 'sphered'

    if np.mean(1-np.abs(x)) > 10**(-eps):  
        #x = x/np.abs(x)
        flag = 'unsphered'

    nx = x/np.abs(x)

    print('finale ', '\t\t|', "%10.3e"%(np.sum(1-np.real(nx*np.conjugate(y))) + lam*np.sum(1 - np.real(nx[0:N-1,:]*np.conjugate(nx[1:N,:]))) + lam*np.sum(1 - np.real(nx[:,0:N-1]*np.conjugate(nx[:,1:N])))), '\t\t\t\t|', flag)

    r = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(r+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        r = r + 1
    print(r, data[r], datatime[r] - starttime)

    if circ==0:
        datei = open('data_S1_2Dgrid_pmm.txt','a')
        for w in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(data[w]))
            datei.write('\n')
        datei.write(str(np.linalg.norm(y0 - x)))
        datei.write('\n')
        datei.write(str(np.sum(np.abs(y0 - x)**2)))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(np.real(nx*np.conj(y0)))**2)))
        datei.write('\n')
        datei.close()
    
    if circ==1:
        datei = open('data_S1_2Dgrid_modpmm.txt','a')
        for w in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(data[w]))
            datei.write('\n')
        datei.write(str(np.linalg.norm(y0 - x)))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(np.real(nx*np.conj(y0))))))
        datei.write('\n')
        datei.close()

    return [nx, data, datares1, datares2, datatime]

def PMM_S2S3_2D(Noise, Data, lam, iter, tau, rho, circ, eps):

    N = np.size(Noise[0,:,0]) 

    a = Noise[0,:,:]
    if a[0,0] == 0:
        dim = 3
    else:
        dim = 4
    b = Noise[1,:,:]
    c = Noise[2,:,:]
    d = Noise[3,:,:]

    aa = Data[0,:,:]
    bb = Data[1,:,:]
    cc = Data[2,:,:]
    dd = Data[3,:,:]

    w = np.zeros((N, N), dtype='float64')
    x = np.zeros((N, N), dtype='float64')
    y = np.zeros((N, N), dtype='float64')
    z = np.zeros((N, N), dtype='float64')

    l1 = np.zeros((N-1,N), dtype='float64')
    l2 = np.zeros((N,N-1), dtype='float64')

    f1 = np.zeros((N-1,N), dtype='float64')
    f2 = np.zeros((N,N-1), dtype='float64')

    g1 = np.zeros((N-1,N), dtype='float64')
    g2 = np.zeros((N,N-1), dtype='float64')

    h1 = np.zeros((N-1,N), dtype='float64')
    h2 = np.zeros((N,N-1), dtype='float64')

    U1 = np.zeros((N-1,N,6,6), dtype=complex)
    U2 = np.zeros((N,N-1,6,6), dtype=complex)

    Id1 = np.zeros((N-1,N,6,6), dtype=complex)
    Id2 = np.zeros((N,N-1,6,6), dtype=complex)
    
    for k in range(6):
        Id1[:,:,k,k] = 1 + 0*1j
        Id2[:,:,k,k] = 1 + 0*1j

    # ||L||^2 6 instead of 2 times the maximum of edges per nodes
    #sigma = 1/(2*2*3*tau)
    sigma = 1/(16*tau)

    k = 0

    data = np.zeros(iter)
    datatime = np.zeros(iter, dtype=float)
    flagggg = 0

    print('iteration \t| func-value \t| original-cost \t| spherical-error')
    print('---------------------------------------------------------------------------')

    starttime = time.time()
    
    #print(np.sum(1 - w*a - x*b - y*c - z*d) + lam*np.sum(1 - w[0:N-1,:]*w[1:N,:] - x[0:N-1,:]*x[1:N,:] - y[0:N-1,:]*y[1:N,:] - z[0:N-1,:]*z[1:N,:]) + lam*np.sum(1 - x[:,0:N-1]*x[:,1:N] -  w[:,0:N-1]*w[:,1:N] - y[:,0:N-1]*y[:,1:N] - z[:,0:N-1]*z[:,1:N]))

    for i in range(0,iter):
        
        [adj_w, adj_x, adj_y, adj_z, adj_l1, adj_l2, adj_f1, adj_f2, adj_g1, adj_g2, adj_h1, adj_h2] = adjopL(U1, U2, N)

        adj_w = adj_w - a
        adj_x = adj_x - b
        adj_y = adj_y - c
        adj_z = adj_z - d

        adj_l1 = adj_l1 - lam
        adj_l2 = adj_l2 - lam

        #adj_e = adj_e - lam
        #adj_f = adj_f - lam    #especially +/-0 since e,f,g not included in f(.) = <c,.>
        #adj_g = adj_g - lam

        if circ == 0:

            [X1, X2] = opL(w - 2*tau*adj_w, x - 2*tau*adj_x, y - 2*tau*adj_y, z - 2*tau*adj_z, l1 - 2*tau*adj_l1, l2 - 2*tau*adj_l2, f1 - 2*tau*adj_f1, f2 - 2*tau*adj_f2, g1 - 2*tau*adj_g1, g2 - 2*tau*adj_g2, h1 - 2*tau*adj_h1, h2 - 2*tau*adj_h2, N)

            [U1, U2] = dim_prox(U1 + sigma*(X1 + Id1), U2 + sigma*(X2 + Id2), N)

            w = w - rho*tau*adj_w
            x = x - rho*tau*adj_x
            y = y - rho*tau*adj_y
            z = z - rho*tau*adj_z

        if circ == 1:

            ow = w - 2*rho*tau*adj_w
            ox = x - 2*rho*tau*adj_x
            oy = y - 2*rho*tau*adj_y
            oz = z - 2*rho*tau*adj_z

            onorm = np.sqrt(ow**2 + ox**2 + oy**2 + oz**2)

            [X1, X2] = opL(ow/onorm, ox/onorm, oy/onorm, oz/onorm, l1 - 2*tau*adj_l1, l2 - 2*tau*adj_l2, f1 - 2*tau*adj_f1, f2 - 2*tau*adj_f2, g1 - 2*tau*adj_g1, g2 - 2*tau*adj_g2, h1 - 2*tau*adj_h1, h2 - 2*tau*adj_h2, N)

            [U1, U2] = dim_prox(U1 + sigma*(X1 + Id1), U2 + sigma*(X2 + Id2), N)

            w = w - rho*tau*adj_w
            x = x - rho*tau*adj_x
            y = y - rho*tau*adj_y
            z = z - rho*tau*adj_z

            norm = np.sqrt(w**2+x**2+y**2+z**2)

            w = w/norm
            x = x/norm
            y = y/norm
            z = z/norm

        l1 = l1 - rho*tau*adj_l1
        f1 = f1 - rho*tau*adj_f1
        g1 = g1 - rho*tau*adj_g1
        h1 = h1 - rho*tau*adj_h1

        l2 = l2 - rho*tau*adj_l2
        f2 = f2 - rho*tau*adj_f2
        g2 = g2 - rho*tau*adj_g2
        h2 = h2 - rho*tau*adj_h2



        norm = np.sqrt(w**2+x**2+y**2+z**2)
        flag = 'sphered'

        data[i] = np.sum(1 - w*a - x*b - y*c - z*d) + lam*np.sum(1 - w[0:N-1,:]*w[1:N,:] - x[0:N-1,:]*x[1:N,:] - y[0:N-1,:]*y[1:N,:] - z[0:N-1,:]*z[1:N,:]) + lam*np.sum(1 - x[:,0:N-1]*x[:,1:N] -  w[:,0:N-1]*w[:,1:N] - y[:,0:N-1]*y[:,1:N] - z[:,0:N-1]*z[:,1:N])

        datatime[i] = time.time()

        if np.mod(i,100) == 0:
            if np.linalg.norm(norm-np.ones(N))>np.exp(-6):
                flag = 'unsphered'

            print(i, '\t\t| ', "%10.3e"%(data[i]), '\t| ', "%10.3e"%(np.sum(1 - w*aa - x*bb - y*cc - z*dd) + lam*np.sum(1 - w[0:N-1,:]*w[1:N,:] - x[0:N-1,:]*x[1:N,:] - y[0:N-1,:]*y[1:N,:] - z[0:N-1,:]*z[1:N,:]) + lam*np.sum(1 - x[:,0:N-1]*x[:,1:N] -  w[:,0:N-1]*w[:,1:N] - y[:,0:N-1]*y[:,1:N] - z[:,0:N-1]*z[:,1:N])), '\t\t| ', "%10.3e"%(1 - np.mean(norm)))


    norm = np.sqrt(w**2+x**2+y**2+z**2)

    diff_W = w - aa
    diff_X = x - bb
    diff_Y = y - cc
    diff_Z = z - dd

    nw = w/norm
    nx = x/norm
    ny = y/norm
    nz = z/norm

    if circ == 0:
        datei = open('data_S{0}_2Dgrid_pmm.txt'.format(dim-1),'a')
        for r in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(np.real(data[r])))
            datei.write('\n')
        datei.write(str(np.sqrt(np.sum(diff_W**2+diff_X**2+diff_Y**2+diff_Z**2))))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(nw*aa + nx*bb + ny*cc + nz*dd))))
        datei.write('\n')
        datei.close()

    if circ == 1:
        datei = open('data_S{0}_2Dgrid_modpmm.txt'.format(dim-1),'a')
        for r in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(np.real(data[r])))
            datei.write('\n')
        datei.write(str(np.sqrt(np.sum(diff_W**2+diff_X**2+diff_Y**2+diff_Z**2))))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(nw*aa + nx*bb + ny*cc + nz*dd))))
        datei.write('\n')
        datei.close()
    
    w = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(w+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        w = w + 1
    print(w, data[w], datatime[w] - starttime)

    return [w, x, y, z, data]


def baseline(y, y0, lam, iter, circ):

    N = np.size(y[0,:,0]) 
    d = np.size(y[:,0,0])

    x = np.zeros((d,N,N))

    print(np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - np.sum(x[:,0:N-1,:]*x[:,1:N,:],0)) + lam*np.sum(1 - np.sum(x[:,:,0:N-1]*x[:,:,1:N],0)))

    data = np.zeros(iter)
    datatime = np.zeros(iter)

    print('iteration \t| func-value \t| original-cost \t| solution is')
    print('-------------------------------------------------------------------')

    starttime = time.time()

    for i in range(iter):

        for r in range(d):
            g = x[r,:,:] - y[r,:,:] + lam*grad(x[r,:,:])
            x[r,:,:] = x[r,:,:] - g/(8*lam + 1)

        data[i] = np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - np.sum(x[:,0:N-1,:]*x[:,1:N,:],0)) + lam*np.sum(1 - np.sum(x[:,:,0:N-1]*x[:,:,1:N],0))

        datatime[i] = time.time()

        norm = np.zeros(N)
        for j in range(d):
            norm = norm + x[j,:,:]**2 
        norm = np.sqrt(norm) 

        flag = 'unsphered'

        for j in range(N):
            for k in range(N):
                if 1 - norm[j,k] > 1e-6:
                    if circ == 1:
                        x[:,j,k] = x[:,j,k]/norm[j,k]
                else:
                    flag = 'sphered'

        if np.mod(i,50)==0:

            print( i, '\t\t| ', "%10.3e"%(data[i]), '\t| ', "%10.3e"%(np.sum(1 - np.sum(x*y0,0)) + lam*np.sum(1 - np.sum(x[:,0:N-1,:]*x[:,1:N,:],0)) + lam*np.sum(1 - np.sum(x[:,:,0:N-1]*x[:,:,1:N],0))), '\t\t|', flag)

    r = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(r+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), 3) + 0.001 or data[ww] < truncate(np.real(data[iter-1]), 3) - 0.001:
                flagg = 0
        r = r + 1
    print(r, data[r], datatime[r] - starttime)


    norm = np.zeros(N)
    for j in range(d):
        norm = norm + x[j,:,:]**2 
    norm = np.sqrt(norm) 
    nx = np.zeros((d,N,N))
    for i in range(N):
        for j in range(N):
            nx[:,i,j] = x[:,i,j]/norm[i,j]
    
    print('finale', '\t\t|', "%10.3e"%(np.sum(1 - np.sum(nx*y,0)) + lam*np.sum(1 - np.sum(nx[:,0:N-1,:]*nx[:,1:N,:],0)) + lam*np.sum(1 - np.sum(nx[:,:,0:N-1]*nx[:,:,1:N],0))), '\t\t\t| ', flag)


    diffX = x - y0

    datei = open('data_S{0}_2Dgrid_ppa.txt'.format(d-1),'a')
    for w in range(np.size(data)):
        datei.write(str(data[w]))
        datei.write('\n')
    datei.write(str(np.sqrt(np.sum(diffX**2))))
    datei.write('\n')
    datei.write(str(np.sum(diffX**2)))
    datei.write('\n')
    datei.write(str(np.sum(np.arccos(np.sum(nx*y0,0))**2)))
    datei.write('\n')
    datei.close()

    return[nx, data]

def ADMM_S1(y, y0, iter, lam, rho, eps):

    N = np.size(y[:,0]) 

    x = np.zeros((N, N), dtype='cdouble')
    r1 = x[0:N-1,:]*np.conjugate(x[1:N,:])
    s2 = np.zeros((N-1,N))
    r2 = x[:,0:N-1]*np.conjugate(x[:,1:N])
    s2 = np.zeros((N,N-1))

    U1 = np.zeros((N-1, N, 3,3), dtype='cdouble')
    U2 = np.zeros((N, N-1, 3,3), dtype='cdouble')

    Z1 = np.zeros((N-1, N, 3,3), dtype='cdouble')
    Z2 = np.zeros((N, N-1, 3,3), dtype='cdouble')

    data = np.zeros(iter)
    datares1 = np.zeros(iter)
    datares2 = np.zeros(iter)
    datatime = np.zeros(iter)
    flagggg = 0

    print('iteration \t| func-value \t| non-convex-cost \t| spherical-error')
    print('------------------------------------------------------------------------')

    starttime = time.time()

    for i in range(iter):
        
        [ux, ur1, ur2] = adjL(U1, U2, N)
        [zx, zr1, zr2] = adjL(Z1, Z2, N)

        x[1:N-1,0] = 1/6*(ux[1:N-1,0] - zx[1:N-1,0] + 1/rho*y[1:N-1,0])
        x[1:N-1,N-1] = 1/6*(ux[1:N-1,N-1] - zx[1:N-1,N-1] + 1/rho*y[1:N-1,N-1])

        x[0,1:N-1] = 1/6*(ux[0,1:N-1] - zx[0,1:N-1] + 1/rho*y[0,1:N-1])
        x[N-1,1:N-1] = 1/6*(ux[N-1,1:N-1] - zx[N-1,1:N-1] + 1/rho*y[N-1,1:N-1])

        x[1:N-1,1:N-1] = 1/8*(ux[1:N-1,1:N-1] - zx[1:N-1,1:N-1] + 1/rho*y[1:N-1,1:N-1])

        x[0,0] = 1/4*(ux[0,0] - zx[0,0] + 1/rho*y[0,0])
        x[0,N-1] = 1/4*(ux[0,N-1] - zx[0,N-1] + 1/rho*y[0,N-1])
        x[N-1,0] = 1/4*(ux[N-1,0] - zx[N-1,0] + 1/rho*y[N-1,0])
        x[N-1,N-1] = 1/4*(ux[N-1,N-1] - zx[N-1,N-1] + 1/rho*y[N-1,N-1])

        r1 = 1/2*(np.real(ur1) - np.real(zr1) + 1/rho*lam)
        s1 = 1/2*(np.imag(ur1) - np.imag(zr1))
        r2 = 1/2*(np.real(ur2) - np.real(zr2) + 1/rho*lam)
        s2 = 1/2*(np.imag(ur2) - np.imag(zr2))


        [L1, L2] = L(x, r1 + 1j*s1, r2 + 1j*s2, N)

        UU1 = U1
        UU2 = U2

        [U1, U2] = proxADMM(L1 + Z1, L2 + Z2, N)

        Z1 = Z1 + L1 - U1
        Z2 = Z2 + L2 - U2
        
        data[i] = np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - r1) + lam*np.sum(1 - r2)
        res1, res2, res3 = adjL(UU1 - U1, UU2 - U2, N)
        datares1[i] = np.linalg.norm(res1)+np.linalg.norm(res2)+np.linalg.norm(res3)
        datares2[i] = np.linalg.norm(L1 - U1) + np.linalg.norm(L2 - U2)
        datatime[i] = time.time()
        if datares1[i] < 10**(-eps) and datares2[i] < 10**(-eps) and flagggg == 0:
            print('iteration :', i, datatime[i] - starttime)
            flagggg = 1

        if np.mod(i,100)==0:

            print(i, '\t\t| ', "%10.3e"%(data[i]), '\t| ', "%10.3e"%(np.sum(1-np.cos(np.angle(x)-np.angle(y0)))+lam*np.sum(1-np.cos(np.angle(x[0:N-1,:])-np.angle(x[1:N,:])))+lam*np.sum(1-np.cos(np.angle(x[:,0:N-1])-np.angle(x[:,1:N])))), '\t\t| ', "%10.3e"%(1-np.mean(np.abs(x))))


    flag = 'sphered'

    if np.mean(1-np.abs(x)) > 10**(-eps):  
        flag = 'unsphered'

    nx = x/np.abs(x)
    
    print('finale ','\t\t|', "%10.3e"%(np.sum(1-np.real(nx*np.conjugate(y))) + lam*np.sum(1 - np.real(nx[0:N-1,:]*np.conjugate(nx[1:N,:]))) + lam*np.sum(1 - np.real(nx[:,0:N-1]*np.conjugate(nx[:,1:N])))), '\t\t\t\t| ', flag)

    r = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(r+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        r = r + 1
    print(r, data[r], datatime[r] - starttime)

    datei = open('data_S1_2Dgrid_admm.txt','a')
    for w in range(np.size(data)):
		#map(data[w], datei.readlines())
        datei.write(str(data[w]))
        datei.write('\n')
    datei.write(str(np.linalg.norm(y0 - nx)))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(x - y0)**2)))
    datei.write('\n')
    datei.write(str(np.sum(np.arccos(np.real(nx*np.conj(y0)))**2)))
    datei.write('\n')
    datei.close()

    return [x, data, datares1, datares2, datatime-starttime]

def ADMM_red(y, y0, iter, lam, rho, eps):

    N = np.size(y[0,:,0])
    d = np.size(y[:,0,0]) 

    x = np.zeros((d, N, N))

    r1 = x[:,0:N-1,:]*np.conjugate(x[:,1:N,:], dtype='float64')
    r2 = x[:,:,0:N-1]*np.conjugate(x[:,:,1:N], dtype='float64')

    U1 = np.zeros((N-1, N, d+2, d+2), dtype='float64')
    U2 = np.zeros((N, N-1, d+2, d+2), dtype='float64')

    Z1 = np.zeros((N-1, N, d+2, d+2), dtype='float64')
    Z2 = np.zeros((N, N-1, d+2, d+2), dtype='float64')

    data = np.zeros(iter, dtype='float64')
    datares1 = np.zeros(iter, dtype='float64')
    datares2 = np.zeros(iter, dtype='float64')
    datatime = np.zeros(iter)
    flagggg = 0

    print('iteration \t| func-value \t| non-convex-cost \t| spherical-error')
    print('--------------------------------------------------------------------------')

    starttime = time.time()

    #print(np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - np.sum(x[:,0:N-1,:]*x[:,1:N,:],0)) + lam*np.sum(1 - np.sum(x[:,:,0:N-1]*x[:,:,1:N],0)))

    for i in range(iter):
        
        [ux, ur1, ur2] = adjL_red(U1, U2, N, d)
        [zx, zr1, zr2] = adjL_red(Z1, Z2, N, d)

        x[:,1:N-1,0] = 1/6*(ux[:,1:N-1,0] - zx[:,1:N-1,0] + 1/rho*y[:,1:N-1,0])
        x[:,1:N-1,N-1] = 1/6*(ux[:,1:N-1,N-1] - zx[:,1:N-1,N-1] + 1/rho*y[:,1:N-1,N-1])

        x[:,0,1:N-1] = 1/6*(ux[:,0,1:N-1] - zx[:,0,1:N-1] + 1/rho*y[:,0,1:N-1])
        x[:,N-1,1:N-1] = 1/6*(ux[:,N-1,1:N-1] - zx[:,N-1,1:N-1] + 1/rho*y[:,N-1,1:N-1])

        x[:,1:N-1,1:N-1] = 1/8*(ux[:,1:N-1,1:N-1] - zx[:,1:N-1,1:N-1] + 1/rho*y[:,1:N-1,1:N-1])

        x[:,0,0] = 1/4*(ux[:,0,0] - zx[:,0,0] + 1/rho*y[:,0,0])
        x[:,0,N-1] = 1/4*(ux[:,0,N-1] - zx[:,0,N-1] + 1/rho*y[:,0,N-1])
        x[:,N-1,0] = 1/4*(ux[:,N-1,0] - zx[:,N-1,0] + 1/rho*y[:,N-1,0])
        x[:,N-1,N-1] = 1/4*(ux[:,N-1,N-1] - zx[:,N-1,N-1] + 1/rho*y[:,N-1,N-1])

        r1 = 1/2*(ur1 - zr1 + 1/rho*lam)
        r2 = 1/2*(ur2 - zr2 + 1/rho*lam)


        [L1, L2] = L_red(x, r1, r2, N, d)

        UU1 = U1
        UU2 = U2

        [U1, U2] = proxADMM(L1.copy() + Z1, L2.copy() + Z2, N)

        Z1 += L1 - U1
        Z2 += L2 - U2

        data[i] = np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - r1) + lam*np.sum(1 - r2)

        res1, res2, res3 = adjL_red(UU1 - U1, UU2 - U2, N, d)
        datares1[i] = np.linalg.norm(res1)+np.linalg.norm(res2)+np.linalg.norm(res3)
        datares2[i] = np.linalg.norm(L1 - U1) + np.linalg.norm(L2 - U2)
        datatime[i] = time.time()
        if datares1[i] < 10**(-eps) and datares2[i] < 10**(-eps) and flagggg == 0:
            print('iteration :', i, datatime[i] - starttime)
            flagggg = 1

        if np.mod(i,100) == 0:

            print(i, '\t\t| ', "%10.3e"%(data[i]), '\t| ', "%10.3e"%(np.sum(1-np.cos(np.angle(x[0,:,:] + 1j*x[1,:,:])-np.angle(y0[0,:,:] + 1j*y0[1,:,:])))+lam*np.sum(1-np.cos(np.angle(x[0,0:N-1,:]+1j*x[1,0:N-1,:])-np.angle(x[0,1:N,:]+1j*x[1,1:N,:])))+lam*np.sum(1-np.cos(np.angle(x[0,:,0:N-1]+1j*x[1,:,0:N-1])-np.angle(x[0,:,1:N]+1j*x[1,:,1:N])))), '\t\t|', "%10.3e"%(1-np.mean(np.sqrt(np.sum(x**2,0)))))


    flag = 'sphered'

    if 1-np.mean(np.sqrt(np.sum(x**2,0))) > 10**(-eps):  
        print(1-np.mean(np.sqrt(np.sum(x**2,0))))
        flag = 'unsphered'

    xnormed = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            xnormed[i,j] = np.sqrt(np.sum(x[:,i,j]**2))
    nx = np.zeros((N,N))
    nx = x/xnormed

    print('finale', '\t\t|', "%10.3e"%(np.sum(1 - np.sum(nx*y,0)) + lam*np.sum(1 - np.sum(nx[:,0:N-1,:]*nx[:,1:N,:],0)) + lam*np.sum(1 - np.sum(nx[:,:,0:N-1]*nx[:,:,1:N],0))), '\t\t\t\t| ', flag )
    
    r = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(r+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        r = r + 1
    print(r, data[r], datatime[r] - starttime)

    datei = open('data_S{0}_2Dgrid_admm_red.txt'.format(d-1),'a')
    for w in range(np.size(data)):
        datei.write(str(data[w]))
        datei.write('\n')
    datei.write(str(np.linalg.norm(y0 - x)))
    datei.write('\n')
    datei.write(str(np.sum(nx - y0)**2))
    datei.write('\n')
    datei.write(str(np.sum(np.arccos(np.sum(nx*y0,0))**2)))
    datei.write('\n')
    datei.close()

    return [x, nx, data, datares1, datares2, datatime-starttime]

def ADMM_red_torus_2D(y, y_0, lam, rho, iter):

    d, N, N = np.shape(y)
    r = int(d/2)
    M = N-1

    x = np.zeros((d, N, N))
    l1 = np.zeros((r, M, N))
    l2 = np.zeros((r, N, M))
    U1 = np.zeros((r,M,N,4,4))
    U2 = np.zeros((r,N,M,4,4))
    Z1 = np.zeros((r,M,N,4,4))
    Z2 = np.zeros((r,N,M,4,4))

    print('iteration \t| func-value \t| torus-error')
    print('---------------------------------------------------')

    for i in range(iter):

        for k in range(r):
            [adjx, adjl1, adjl2] = adjL_red(U1[k,:,:,:,:] - Z1[k,:,:,:,:], U2[k,:,:,:,:] - Z2[k,:,:,:,:], N, 2)

            #s <- argmin_s f(s) + rho/2*||Ls - u + z||^2  ------- first ADMM-step
            x[2*k:2*k+2,0,0] = 1/4*(adjx[:,0,0] + 1/rho*y[2*k:2*k+2,0,0])
            x[2*k:2*k+2,0,N-1] = 1/4*(adjx[:,0,N-1] + 1/rho*y[2*k:2*k+2,0,N-1])
            x[2*k:2*k+2,N-1,0] = 1/4*(adjx[:,N-1,0] + 1/rho*y[2*k:2*k+2,N-1,0])
            x[2*k:2*k+2,N-1,N-1] = 1/4*(adjx[:,N-1,N-1] + 1/rho*y[2*k:2*k+2,N-1,N-1])

            x[2*k:2*k+2,0,1:N-1] = 1/6*(adjx[:,0,1:N-1] + 1/rho*y[2*k:2*k+2,0,1:N-1])
            x[2*k:2*k+2,1:N-1,0] = 1/6*(adjx[:,1:N-1,0] + 1/rho*y[2*k:2*k+2,1:N-1,0])
            x[2*k:2*k+2,N-1,1:N-1] = 1/6*(adjx[:,N-1,1:N-1] + 1/rho*y[2*k:2*k+2,N-1,1:N-1])
            x[2*k:2*k+2,1:N-1,N-1] = 1/6*(adjx[:,1:N-1,N-1] + 1/rho*y[2*k:2*k+2,1:N-1,N-1])

            x[2*k:2*k+2,1:N-1,1:N-1] = 1/8*(adjx[:,1:N-1,1:N-1] + 1/rho*y[2*k:2*k+2,1:N-1,1:N-1])

            l1[k,:,:] = 1/2*(adjl1 + 1/rho*lam)
            l2[k,:,:] = 1/2*(adjl2 + 1/rho*lam)

            #U <- argmin (.) = prox_{hpsd + I => 0}(.)  ------- second ADMM-step
            temp1, temp2 = L_red(x[2*k:2*k+2,:,:], l1[k,:,:], l2[k,:,:], N, 2)
            #Utemp = U
            [U1[k,:,:,:,:], U2[k,:,:,:,:]] = proxADMM(temp1.copy() + Z1[k,:,:,:,:], temp2.copy() + Z2[k,:,:,:,:], N)

            #Z <- Z + Ls - U  ------- third ADMM-step // update
            Z1[k,:,:,:,:] += temp1 - U1[k,:,:,:,:]
            Z2[k,:,:,:,:] += temp2 - U2[k,:,:,:,:]
        
        norm = np.zeros((r, N, N))
        for w in range(r):
            norm[w,:,:] = np.sum(x[2*w:2*w+2,:,:]**2, 0)
        
        if np.mod(i, 100) == 0:
            print(i, '\t\t|', "%10.2e"% (np.sum(-np.sum(x*y, 0)) - lam*np.sum(l1) - lam*np.sum(l2)), '\t| ', "%10.2e"% np.mean(1 - np.sqrt(norm)))
    
    return [x, l1, l2]

def ADMM_red_hyper2D(y, y_0, lam, rho, iter):

    d, N, N = np.shape(y)

    U1 = np.zeros((d+4,d+4,N-1,N))
    U2 = np.zeros((d+4,d+4,N,N-1))
    Z1 = np.zeros((d+4,d+4,N-1,N))
    Z2 = np.zeros((d+4,d+4,N,N-1))

    x = np.zeros((d,N,N))
    v = np.zeros((N,N))
    l1 = np.zeros(N-1)
    f1 = np.zeros(N-1)
    l2 = np.zeros(N-1)
    f2 = np.zeros(N-1)

    E1 = np.zeros((d+4,d+4,N-1,N))
    for i in range(d):
        E1[i,i,:,:] = np.ones((N-1,N))
    E1[d+1,d,:,:] = -np.ones((N-1,N))
    E1[d,d+1,:,:] = -np.ones((N-1,N))
    E1[d+3,d+2,:,:] = -np.ones((N-1,N))
    E1[d+2,d+3,:,:] = -np.ones((N-1,N))

    E2 = np.zeros((d+4,d+4,N,N-1))
    for i in range(d):
        E2[i,i,:,:] = np.ones((N,N-1))
    E2[d+1,d,:,:] = -np.ones((N,N-1))
    E2[d,d+1,:,:] = -np.ones((N,N-1))
    E2[d+3,d+2,:,:] = -np.ones((N,N-1))
    E2[d+2,d+3,:,:] = -np.ones((N,N-1))

    x1 = np.random.randn(d,N,N)
    i = 0 

    print('iteration \t| func-value \t| mikwosky-error \t| error')
    print('-------------------------------------------------------------------------')

    #for i in range(iter):
    while np.linalg.norm(x1 - x) > 1e-4 and np.linalg.norm(1 + np.sum(x[0:d-1,:,:]**2,0) - x[d-1,:,:]**2) > 1e-4 and i < iter:

        adj_x, adj_v, adj_f1, adj_l1, adj_f2, adj_l2 = adjopL_2DHyper(U1 - Z1, U2 - Z2)

        x1 = x.copy()

        x[:,0,0] = 1/8*(adj_x[:,0,0] + y[:,0,0]/rho)
        x[:,0,N-1] = 1/8*(adj_x[:,0,N-1] + y[:,0,N-1]/rho)
        x[:,N-1,0] = 1/8*(adj_x[:,N-1,0] + y[:,N-1,0]/rho)
        x[:,N-1,N-1] = 1/8*(adj_x[:,N-1,N-1] + y[:,N-1,N-1]/rho)

        x[:,0,1:N-1] = 1/12*(adj_x[:,0,1:N-1] + y[:,0,1:N-1]/rho)
        x[:,N-1,1:N-1] = 1/12*(adj_x[:,N-1,1:N-1] + y[:,N-1,1:N-1]/rho)
        x[:,1:N-1,0] = 1/12*(adj_x[:,1:N-1,0] + y[:,1:N-1,0]/rho)
        x[:,1:N-1,N-1] = 1/12*(adj_x[:,1:N-1,N-1] + y[:,1:N-1,N-1]/rho)

        x[:,1:N-1,1:N-1] = 1/16*(adj_x[:,1:N-1,1:N-1] + y[:,1:N-1,1:N-1]/rho)

        f1 = 1/4*(adj_f1 + lam/rho)
        f2 = 1/4*(adj_f2 + lam/rho)

        v[0,0] = 1/4*(adj_v[0,0] - 1/2/rho - lam/rho)
        v[0,N-1] = 1/4*(adj_v[0,N-1] - 1/2/rho - lam/rho)
        v[N-1,0] = 1/4*(adj_v[N-1,0] - 1/2/rho - lam/rho)
        v[N-1,N-1] = 1/4*(adj_v[N-1,N-1] - 1/2/rho - lam/rho)
        
        v[0,1:N-1] = 1/6*(adj_v[0,1:N-1] - 1/2/rho - 3/2*lam/rho)
        v[1:N-1,0] = 1/6*(adj_v[1:N-1,0] - 1/2/rho - 3/2*lam/rho)
        v[N-1,1:N-1] = 1/6*(adj_v[N-1,1:N-1] - 1/2/rho - 3/2*lam/rho)
        v[1:N-1,N-1] = 1/6*(adj_v[1:N-1,N-1] - 1/2/rho - 3/2*lam/rho)

        v[1:N-1,1:N-1] = 1/8*(adj_v[1:N-1,1:N-1] - 1/2/rho - 2*lam/rho)

        l1 = 1/4*(adj_l1)
        l2 = 1/4*(adj_l2)

        ### --------------

        temp1, temp2 = opL_2DHyper(x, v, f1, l1, f2, l2)

        U1 = prox_2DHyper(temp1.copy() + Z1 + E1) - E1
        U2 = prox_2DHyper(temp2.copy() + Z2 + E2) - E2

        ### --------------

        Z1 += temp1 - U1
        Z2 += temp2 - U2

        if np.mod(i,100) == 0:
            print(i, '\t\t|', "%10.2e"% (np.sum(1/2*(np.sum(y**2,0) + np.sum(x**2,0) - 2*np.sum(x*y,0)) + lam/2*(np.sum(v[0:N-1,:]) + np.sum(v[1:N,:]) - 2*np.sum(f1))) + lam/2*(np.sum(v[:,0:N-1]) + np.sum(v[:,1:N]) - 2*np.sum(f2))), '\t|', "%10.2e"% np.linalg.norm(1 + np.sum(x[0:d-1,:,:]**2,0) - x[d-1,:,:]**2), '\t\t|', "%10.2e"% np.linalg.norm(x1 - x))

        i += 1
        
    return [x, v, f1, l1, f2, l2]

#############################################
#
# plots
#
#############################################

def plotS1_image(Noise, Data, q):
    plt.figure(figsize=(17,5), dpi=200)
    G = matplotlib.gridspec.GridSpec(1, 95)

    level = np.arange(-np.pi,np.pi+0.0001,np.pi*0.05)
    levelcbar = np.arange(-np.pi,np.pi+0.0001,np.pi*0.2)

    ax = plt.subplot(G[0, 0:29])
    ax.imshow(np.angle(Data[0,:,:] + 1j*Data[1,:,:]), cmap=cm.twilight, vmin=-np.pi, vmax=np.pi)
    ax = plt.subplot(G[0, 30:59])
    ax.imshow(np.angle(Noise[0,:,:] + 1j*Noise[1,:,:]), cmap=cm.twilight, vmin=-np.pi, vmax=np.pi)
    ax.set_yticks([])
    ax2 = plt.subplot(G[0, 60:95])
    im = ax2.imshow(np.angle(q[0,:,:] + 1j*q[1,:,:]), cmap=cm.twilight, vmin=-np.pi, vmax=np.pi)

    ax2.set_yticks([])
    plt.colorbar(im)

def plotS1_hsv_image(Noise, Data, q):
    plt.figure(figsize=(17,5), dpi=200)
    G = matplotlib.gridspec.GridSpec(1, 90)

    level = np.arange(-np.pi,np.pi+0.0001,np.pi*0.05)
    levelcbar = np.arange(-np.pi,np.pi+0.0001,np.pi*0.2)

    ax = plt.subplot(G[0, 0:29])
    ax.imshow(Data[:,:,0], cmap=cm.hsv, vmin=0, vmax=180)  #origin='upper'
    ax = plt.subplot(G[0, 30:59])
    ax.imshow(Noise[:,:,0], cmap=cm.hsv, vmin=0, vmax=180)
    ax.set_yticks([])
    ax2 = plt.subplot(G[0, 60:90])
    im = ax2.imshow(q[:,:,0], cmap=cm.hsv, vmin=0, vmax=180)
    ax2.set_yticks([])

def plotS2_image(Noise, Data, q):

    [p0, t0] = angle_S2(Data[0,:,:], Data[1,:,:], Data[2,:,:])
    [p, t] = angle_S2(Noise[0,:,:], Noise[1,:,:], Noise[2,:,:])
    [p1, t1] = angle_S2(q[0,:,:], q[1,:,:], q[2,:,:])

    plt.rc('font', size=10) 

    fig = plt.figure(figsize=(20,6))

    n = np.size(p0[0,:])
    level = np.arange(-np.pi,np.pi+0.0001,np.pi*0.1)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(p0, cmap=cm.twilight, alpha=t0/np.max(t0))
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax1 = fig.add_subplot(1, 3, 2)
    ax1.imshow(p, cmap=cm.twilight, alpha=t/np.max(t))
    ax1.set_yticks([])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.imshow(p1, cmap=cm.twilight, alpha=t1/np.max(t1))
    ax2.set_yticks([])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

def plotS2_rgb_image(Noiseimag, Dataimag, q):

    plt.figure(figsize=(17,5), dpi=200)
    G = matplotlib.gridspec.GridSpec(1, 90)

    level = np.arange(-np.pi,np.pi+0.0001,np.pi*0.05)
    levelcbar = np.arange(-np.pi,np.pi+0.0001,np.pi*0.2)

    ax = plt.subplot(G[0, 0:29])
    ax.imshow(np.transpose(Dataimag, (1,2,0)))  #origin='upper'
    ax = plt.subplot(G[0, 30:59])
    ax.imshow(np.transpose(Noiseimag, (1,2,0)))
    ax.set_yticks([])
    ax2 = plt.subplot(G[0, 59:90])
    im = ax2.imshow(np.transpose(q, (1,2,0)))
    ax2.set_yticks([])

def quad2rotation(x):
    R = np.zeros((3,3))
    R[0,0] = 1 - 2 * x[2]**2 - 2 * x[3]**2
    R[0,1] = 2 * x[1] * x[2] - 2 * x[0] * x[3]
    R[0,2] = 2 * x[1] * x[3] + 2 * x[0] * x[2]
    R[1,0] = 2 * x[1] * x[2] + 2 * x[0] * x[3]
    R[1,1] = 1 - 2 * x[1]**2 - 2 * x[3]**2
    R[1,2] = 2 * x[2] * x[3] - 2 * x[0] * x[1]
    R[2,0] = 2 * x[1] * x[3] - 2 * x[0] * x[2]
    R[2,1] = 2 * x[2] * x[3] + 2 * x[0] * x[1]
    R[2,2] = 1 - 2 * x[1]**2 - 2 * x[2]**2
    return R

def createCone(l, r, n=10, m=11):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(-1, 1, m)
    X = np.zeros((n,m,3))
    X[:,:,0] = np.outer(r * np.cos(u), r * l * (1 - np.abs(v)))
    X[:,:,1] = np.outer(r * np.sin(u), r * l * (1 - np.abs(v)))
    X[:,:,2] = np.outer(np.ones(np.size(u)), np.maximum(l * v, 0)) - l / 2
    cval = np.outer(np.linspace(0,1,n), np.ones(np.size(v)))
    c = cm.twilight(cval)
    return (X,c)

def rotateMesh(X, q):
    n1 = X.shape[0]
    n2 = X.shape[1]
    R = quad2rotation(q)
    Y = np.zeros_like(X)
    for k1 in range(n1):
        for k2 in range(n2):
            Y[k1,k2,:] = R @ X[k1,k2,:]
    return Y 

def plotSO3(ax, x, l=0.8, r=0.5):
    n1 = x.shape[0]
    n2 = x.shape[1]
    (X,c) = createCone(l, r)
    for k1 in range(n1):
        for k2 in range(n2):
            Y = rotateMesh(X, x[k1,k2])
            mx = Y[:,:,0] + k1
            my = Y[:,:,1] + k2
            mz = Y[:,:,2]
            ax.plot_surface(mx, my, mz, facecolors = c, lightsource=LightSource(altdeg=-85))

def plotSO3_image_cones(Noise, Data, qq):

    xnew = np.arange(25,75,3)
    xpoints, ypoints = np.meshgrid(xnew, xnew)

    plt.rc('font', size=10) 

    fig = plt.figure(figsize=(3*15,15))

    ax = fig.add_subplot(1,3,1, projection='3d', position=(0.01, 0.99, 0.01, 0.99))
    plotSO3(ax, np.transpose(Data[:, xpoints, ypoints], (1,2,0)))
    ax.view_init(elev=90, azim=0, roll=0)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    ax.set_zticks([])
    ax.set_aspect('equal')

    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.set_yticks([])
    plotSO3(ax1, np.transpose(Noise[:,xpoints, ypoints], (1,2,0)))
    ax1.view_init(elev=90, azim=0, roll=0)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    ax1.set_zticks([])
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    ax2.set_yticks([])
    plotSO3(ax2, np.transpose(qq[:,xpoints, ypoints], (1,2,0)))
    ax2.view_init(elev=90, azim=0, roll=0)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    ax2.set_zticks([])
    ax2.set_aspect('equal')
    plt.tight_layout(h_pad=-35, w_pad=-35)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

def plotSO3_image(Noise, Data, qq):

    [a0, p0, t0] = angle_SO3(Data)
    [a, p, t] = angle_SO3(Noise)
    [a1, p1, t1] = angle_SO3(qq)

    plt.rc('font', size=10) 

    fig = plt.figure(figsize=(20,6))

    n = np.size(a0[0,:])

    ax = fig.add_subplot(1, 3, 1)
    for i in range(n):
        for j in range(n):
            ax.plot(i,j,'ko', markersize=1.5 , color=lighten_color([(p0[i,j]-np.min(p0))/(np.max(p0)-np.min(p0)),1.0,0.0], 0.5+1.*(t0[i,j]-np.min(t0))/(np.max(t0)-np.min(t0))), label=['phi', 'theta'])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax1 = fig.add_subplot(1, 3, 2)
    for i in range(n):
        for j in range(n):
            ax1.plot(i,j, 'ko', markersize=1.5 , color=lighten_color([(p[i,j]-np.min(p))/(np.max(p)-np.min(p)),1.0,0.0], 0.5+1.*(t[i,j]-np.min(t))/(np.max(t)-np.min(t))), label=['phi', 'theta'])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax2 = fig.add_subplot(1, 3, 3)
    for i in range(n):
        for j in range(n):
            ax2.plot(i,j, 'ko', markersize=1.5 , color=lighten_color([(p1[i,j]-np.min(p1))/(np.max(p1)-np.min(p1)),1.0,0.0], 0.5+1.*(t1[i,j]-np.min(t1))/(np.max(t1)-np.min(t1))), label=['phi', 'theta'])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

def draw_line(x, y, angle1, angle2):
    r = 1  # or whatever fits you
    plt.arrow(x, y, r*np.cos(angle1), r*np.sin(angle1), head_starts_at_zero=True, color=plt.get_cmap('twilight')(angle2))

def draw_line_non_periodic(x, y, angle1, angle2):
    r = 1  # or whatever fits you
    plt.arrow(x, y, r*np.cos(angle1), r*np.sin(angle1), head_starts_at_zero=True, color=plt.get_cmap('YlGnBu')(angle2))

def plotTorus(Noise, Data, sol_x):
    
    fig = plt.figure(figsize=(12,4), dpi=100)
    ax = fig.add_subplot(131,frameon=False)

    for i in range(np.size(Data[0,:,0])):
        for j in range(np.size(Data[0,0,:])):
            draw_line(i, j, np.angle(Data[0,i,j] + 1j*Data[1,i,j]), np.angle(Data[2,i,j] + 1j*Data[3,i,j]))
        
    ax.axis('off')

    ax = fig.add_subplot(132,frameon=False)

    for i in range(np.size(Noise[0,:,0])):
        for j in range(np.size(Noise[0,0,:])):
            draw_line(i, j, np.angle(Noise[0,i,j] + 1j*Noise[1,i,j]), np.angle(Noise[2,i,j] + 1j*Noise[3,i,j]))

    ax.axis('off')

    ax = fig.add_subplot(133,frameon=False)

    for i in range(np.size(sol_x[0,:,0])):
        for j in range(np.size(sol_x[0,0,:])):
            draw_line(i, j, np.angle(sol_x[0,i,j] + 1j*sol_x[1,i,j]), np.angle(sol_x[2,i,j] + 1j*sol_x[3,i,j]))

    ax.axis('off')
    plt.tight_layout()

def plot_hyper1(Noise, Data, sol_x):

    fig = plt.figure(figsize=(15,5))

    ax1 = fig.add_subplot(131)
    a1 = ax1.imshow(np.arcsinh(Data[0,:,:]))
    fig.colorbar(a1, ax = ax1, location='right', shrink=0.76)

    ax2 = fig.add_subplot(132)
    a2 = ax2.imshow(np.arcsinh(Noise[0,:,:]))
    fig.colorbar(a2, ax = ax2, location='right', shrink=0.76)

    ax3 = fig.add_subplot(133)
    a3 = ax3.imshow(np.arcsinh(sol_x[0,:,:]))
    fig.colorbar(a3, ax = ax3, location='right', shrink=0.76)
    fig.tight_layout()
    #fig.savefig('1-hyperboloid_denoising_sig=0.3.pdf', dpi=300)

def plot_hyper2(Noise, Data, sol_x):

    rNoise = np.arccosh(Noise[2,:,:])
    rData = np.arccosh(Data[2,:,:])
    rsol_x = np.arccosh(sol_x[2,:,:])

    aNoise = np.arcsin(Noise[0,:,:]/np.sinh(rNoise))
    aData = np.arcsin(Data[0,:,:]/np.sinh(rData))
    asol_x = np.arcsin(sol_x[0,:,:]/np.sinh(rsol_x))
    
    fig = plt.figure(figsize=(12,4), dpi=100)
    ax = fig.add_subplot(131,frameon=False)

    for i in range(np.size(Data[0,:,0])):
        for j in range(np.size(Data[0,0,:])):
            draw_line_non_periodic(i, j, aData[i,j], rData[i,j])
        
    ax.axis('off')

    ax = fig.add_subplot(132,frameon=False)

    for i in range(np.size(Noise[0,:,0])):
        for j in range(np.size(Noise[0,0,:])):
            draw_line_non_periodic(i, j, aNoise[i,j], rNoise[i,j])

    ax.axis('off')

    ax = fig.add_subplot(133,frameon=False)

    for i in range(np.size(sol_x[0,:,0])):
        for j in range(np.size(sol_x[0,0,:])):
            draw_line_non_periodic(i, j, asol_x[i,j], rsol_x[i,j])

    ax.axis('off')
    plt.tight_layout()
    #fig.savefig('2-hyperboloid_denoising_image_sig_0.6.pdf',dpi=300)
