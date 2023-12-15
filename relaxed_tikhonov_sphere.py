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
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
from mpl_toolkits.mplot3d import Axes3D


def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def reconstruction(x):

    N = np.size(x)
    for j in range(3):
        for i in range(1,N):
            if np.abs(np.abs(x[i] - x[i-1])) > 0.9*np.pi:
                if x[i] - x[i-1] < 0:
                    x[i] = x[i] + 2*np.pi
                    #if x[i-1] < -2*np.pi:
                    #    x[i] = x[i] - 2*np.pi
                else:
                    x[i] = x[i] - 2*np.pi

    return x

def angle_S2(a, b, c):

    v = np.array([a, b, c]/np.sqrt(a**2 + b**2 + c**2))
    theta = np.arccos(v[2])
    phi = np.arctan2(c, b)

    return [phi, theta]

def angle_S3(Data):


    alpha = np.arctan2(np.sqrt(Data[1,:]**2+Data[2,:]**2+Data[3,:]**2), Data[0,:])
    v = np.array([Data[1,:], Data[2,:], Data[3,:]]/np.sqrt(Data[1,:]**2+Data[2,:]**2+Data[2,:]**2))
    theta = np.arccos(v[2])
    phi = np.arctan2(Data[2,:], Data[1,:])

    return [alpha, phi, theta]

def angle_SO3(Data):

    for i in range(np.size(Data[0,:])):
        if Data[0,i]<0:
            Data[:,i] = -Data[:,i]
            
    alpha = np.arctan2(np.sqrt(Data[1,:]**2+Data[2,:]**2+Data[3,:]**2), Data[0,:])
    v = np.array([Data[1,:], Data[2,:], Data[3,:]]/np.sqrt(Data[1,:]**2+Data[2,:]**2+Data[3,:]**2))
    theta = np.arccos(v[2])
    phi = np.arctan2(Data[2,:], Data[1,:])

    return [alpha, phi, theta]

def back_SO3(a,p,t):

    N = np.size(a)
    w = np.zeros((4,N))

    w[0,:] = np.sin(p)*np.sin(t)*np.sin(a)
    w[1,:] = np.cos(p)*np.sin(t)*np.sin(a)
    w[2,:] = np.cos(t)*np.sin(a)
    w[3,:] = np.cos(a)

    return w

def sig(x):

    x = np.mod(x, 2*np.pi)

    x = np.where(x > np.pi, x-2*np.pi, x)
    
    return x

def sig2(x):

    x = np.mod(x, np.pi)
    
    return x

def signal(x):

    if 0 <= x <= 1/4:
        return sig(-24*np.pi*x**2 + 3/4*np.pi)
    if 0.25 < x <= 3/8:
        return sig(4*np.pi*x - np.pi/4)
    if 3/8 < x <= 1/2:
        return sig(-np.pi*x - 3/8)
    for j in range(4):
        if (3*j + 16)/32 < x <= (3*j + 19)/32:
            return sig(-(j+7)/8*np.pi)
    if 7/8 < x <= 1:
        return 3/2*np.pi*np.exp(-1/7 - 1/(1-x)) - 3/4*np.pi

def signal0(x):

    if 0 <= x <= 1/4:
        return np.pi-0.75
    if 0.25 < x <= 3/8:
        return np.pi-1.5
    if 3/8 < x <= 1/2:
        return np.pi-2.5
    for j in range(4):
        if (3*j + 16)/32 < x <= (3*j + 19)/32:
            return j/4*(np.pi-0.1) + (1 - j/4)*(0.1)
    if 7/8 < x <= 1:
        return 3/4*np.pi

def sample_toy_signal_S2(N, lam):

    line = np.arange(N)/(N-1)

    s_01 = np.array([signal(line[i]) for i in range(N)])
    s_02 = np.array([signal0(line[i]) for i in range(N)])

    noise = lam*np.random.randn(N)
    noise0 = lam*np.random.randn(N)

    s_n1 = s_01 + noise
    s_n2 = s_02 + noise0
    s_n1 = np.array([sig(s_n1[i]) for i in range(N)])
    s_n2 = np.array([sig2(s_n2[i]) for i in range(N)])

    plt.figure(0,figsize=(15,3), dpi=200)
    plt.plot(line, s_01, 'b')
    plt.plot(line, s_n1, 'k', linewidth=0.5)
    plt.figure(1,figsize=(15,3), dpi=200)
    plt.plot(line, s_02, 'b')
    plt.plot(line, s_n2, 'k', linewidth=0.5)

    s_0 = np.array((np.cos(s_02)*np.sin(s_01), np.sin(s_02)*np.sin(s_01), np.cos(s_01)))
    s_n = np.array((np.cos(s_n2)*np.sin(s_n1), np.sin(s_n2)*np.sin(s_n1), np.cos(s_n1)))

    return [s_n, s_0]

def sample_vMF_signal(n,d,kappa):

    ''' sphere-valued data  | smooth ground truth, with sND(., 1/lam) ~ vMF(., lam) in direction (0,0,1) 
                            | - use finer grid (10x - see circle-valued data) with noide-parameter kap in O(10**-1)
                            |       -> noise apears with koeffizent kap*1/10*sqrt(lam)'''

    N = 20*n
    kap = 1

    X = np.zeros((d,n))
    X[d-1,0] = 1
    for i in range(1,n):
        X[:,i] = X[:,i-1] + kap*np.random.randn(d)

    xx = np.linspace(0,20*n,n)
    xxnew = np.linspace(0, 20*n, 20*n)

    XX = np.zeros((d,N))

    for i in range(d):
        spl = CubicSpline(xx, X[i,:])
        XX[i,:] = spl(xxnew)

    eX = np.array([XX[:,i]/np.linalg.norm(XX[:,i]) for i in range(N)]).T

    eY = [vonmises_fisher(mu=eX[:,i], kappa=kappa).rvs(1) for i in range(N)]

    print('normalization test : ', np.mean([np.linalg.norm(eY[i]) for i in range(N)]))

    if d == 2:
        X01 = np.array([eX[0,i] for i in range(N)])
        X1 = np.array([eY[i][0][0] for i in range(N)])
        Y01 = np.array([eX[1,i] for i in range(N)])
        Y1 = np.array([eY[i][0][1] for i in range(N)])

        return [np.array([X1, Y1]), np.array([X01, Y01])]

    if d == 3:

        X01 = np.array([eX[0,i] for i in range(N)])
        X1 = np.array([eY[i][0][0] for i in range(N)])
        Y01 = np.array([eX[1,i] for i in range(N)])
        Y1 = np.array([eY[i][0][1] for i in range(N)])
        Z01 = np.array([eX[2,i] for i in range(N)])
        Z1 = np.array([eY[i][0][2] for i in range(N)])

        return [np.array([X1, Y1, Z1]), np.array([X01, Y01, Z01])]
    
    if d == 4:

        W01 = np.array([eX[0,i] for i in range(N)])
        W1 = np.array([eY[i][0][0] for i in range(N)])
        X01 = np.array([eX[1,i] for i in range(N)])
        X1 = np.array([eY[i][0][1] for i in range(N)])
        Y01 = np.array([eX[2,i] for i in range(N)])
        Y1 = np.array([eY[i][0][2] for i in range(N)])
        Z01 = np.array([eX[3,i] for i in range(N)])
        Z1 = np.array([eY[i][0][3] for i in range(N)])

        return [np.array([W1, X1, Y1, Z1]), np.array([W01, X01, Y01, Z01])]
    
    if d == 8:

        W01 = np.array([eX[0,i] for i in range(N)])
        W1 = np.array([eY[i][0][0] for i in range(N)])
        X01 = np.array([eX[1,i] for i in range(N)])
        X1 = np.array([eY[i][0][1] for i in range(N)])
        Y01 = np.array([eX[2,i] for i in range(N)])
        Y1 = np.array([eY[i][0][2] for i in range(N)])
        Z01 = np.array([eX[3,i] for i in range(N)])
        Z1 = np.array([eY[i][0][3] for i in range(N)])
        R01 = np.array([eX[4,i] for i in range(N)])
        R1 = np.array([eY[i][0][4] for i in range(N)])
        S01 = np.array([eX[5,i] for i in range(N)])
        S1 = np.array([eY[i][0][5] for i in range(N)])
        T01 = np.array([eX[6,i] for i in range(N)])
        T1 = np.array([eY[i][0][6] for i in range(N)])
        U01 = np.array([eX[7,i] for i in range(N)])
        U1 = np.array([eY[i][0][7] for i in range(N)])

        return [np.array([W1, X1, Y1, Z1, R1, S1, T1, U1]), np.array([W01, X01, Y01, Z01, R01, S01, T01, U01])]


def sample_vMF_noise(signal):

    ''' sphere-valued data  | smooth ground truth, with sND(., 1/lam) ~ vMF(., lam) in direction (0,0,1) 
                            | - use finer grid (10x - see circle-valued data) with noide-parameter kap in O(10**-1)
                            |       -> noise apears with koeffizent kap*1/10*sqrt(lam)'''

    d, N = np.shape(signal)

    eY = [vonmises_fisher(mu=signal[:,i], kappa=10).rvs(1) for i in range(N)]

    print('normalization test : ', np.mean([np.linalg.norm(eY[i]) for i in range(N)]))

    if d == 2:
        X1 = np.array([eY[i][0][0] for i in range(N)])
        Y1 = np.array([eY[i][0][1] for i in range(N)])

        return np.array([X1, Y1])

    if d == 3:

        X1 = np.array([eY[i][0][0] for i in range(N)])
        Y1 = np.array([eY[i][0][1] for i in range(N)])
        Z1 = np.array([eY[i][0][2] for i in range(N)])

        return np.array([X1, Y1, Z1])
    
def sample_SO3_signal_vMF(n):

    ''' sphere-valued data  | smooth ground truth, with sND(., 1/lam) ~ vMF(., lam) in direction (0,0,1) 
                        | - use finer grid (10x - see circle-valued data) with noide-parameter kap in O(10**-1)
                        |       -> noise apears with koeffizent kap*1/10*sqrt(lam)'''

    N = 20*n
    kap1 = 1
    kap2 = 0.5
    sig = 0.25
    lam_vMF = 1000

    X = np.zeros((3,n))
    X[2,0] = 1
    for i in range(1,n):
        X[:,i] = X[:,i-1] + kap1*np.random.randn(3)

    xx = np.linspace(0,20*n,n)
    xxnew = np.linspace(0, 20*n, 20*n)

    XX = np.zeros((3,N))

    for i in range(3):
        spl = CubicSpline(xx, X[i,:])
        XX[i,:] = spl(xxnew)

    eX = np.array([XX[:,i]/np.linalg.norm(XX[:,i]) for i in range(N)]).T

    eY = np.array([vonmises_fisher(mu=eX[:,i], kappa=30).rvs(1) for i in range(N)])

    print('normalization test : ', np.mean([np.linalg.norm(eY[i]) for i in range(N)]))

    A = np.zeros((2,n))
    A[1,0] = 1
    for i in range(1,n):
        A[:,i] = A[:,i-1] + kap1*np.random.randn(2)/np.pi

    aa = np.linspace(0,20*n,n)
    aanew = np.linspace(0, 20*n, 20*n)

    AA = np.zeros((2,N))

    for i in range(2):
        spl = CubicSpline(aa, A[i,:])
        AA[i,:] = spl(aanew)

    NA = np.array([AA[:,i]/np.linalg.norm(AA[:,i]) for i in range(N)]).T
    NAA = np.array([vonmises_fisher(mu=NA[:,i], kappa=15).rvs(1) for i in range(N)])

    WAA = np.angle(NA[0,:] + 1j*NA[1,:])
    WNA = np.angle(NAA[:,0,0] + 1j*NAA[:,0,1])

    eX = eX*np.sin(WAA/2)
    eY = eY[:,0,:]
    eY = eY.T*np.sin(WNA/2)


    W01 = np.cos(WAA/2)
    W1 = np.cos(WNA/2)
    X01 = np.array([eX[0,i] for i in range(N)])
    X1 = np.array([eY[0,i] for i in range(N)])
    Y01 = np.array([eX[1,i] for i in range(N)])
    Y1 = np.array([eY[1,i] for i in range(N)])
    Z01 = np.array([eX[2,i] for i in range(N)])
    Z1 = np.array([eY[2,i] for i in range(N)])

    print('normalization test : ', np.mean([np.linalg.norm(np.array([W1[i] , X1[i], Y1[i], Z1[i]])) for i in range(N)]))

    return [np.array([W1, X1, Y1, Z1]), np.array([W01, X01, Y01, Z01])]

def transformation_SO3(Data):

    n = np.size(Data[0,:])
    XX = Data
    d = []

    for i in range(n-1):
        if np.sum(XX[:,i]*XX[:,i+1]) < 0:
            XX[:,i+1] = -XX[:,i+1]
            d = np.append(d, i+1)

    return [XX, d]

def sample_torus_vMF_signal(d, n, kap):

    xxnew = np.linspace(0, 20*n, 20*n)

    Data = np.zeros((2*d, np.size(xxnew)))
    Noise = np.zeros((2*d, np.size(xxnew)))

    for i in range(d):
        noise, data = sample_vMF_signal(n, 2, kap)
        Data[2*i: 2*i+2, :] = data
        Noise[2*i: 2*i+2, :] = noise
    
    return [Noise, Data]

def back_transformation_SO3(Data, d):

    for i in d:
        Data[:,int(i)] = -Data[:,int(i)]
    
    return Data


def angle_S7(X):

    a = X[0,:] 
    b = X[1,:] 
    c = X[2,:] 
    d = X[3,:] 
    e = X[4,:] 
    f = X[5,:] 
    g = X[6,:] 
    h = X[7,:]

    phi1 = np.arccos(a/np.sqrt(a**2+b**2+c**2+d**2+e**2+f**2+g**2+h**2))
    phi2 = np.arccos(b/np.sqrt(b**2+c**2+d**2+e**2+f**2+g**2+h**2))
    phi3 = np.arccos(c/np.sqrt(c**2+d**2+e**2+f**2+g**2+h**2))
    phi4 = np.arccos(d/np.sqrt(d**2+e**2+f**2+g**2+h**2))
    phi5 = np.arccos(e/np.sqrt(e**2+f**2+g**2+h**2))
    phi6 = np.arccos(f/np.sqrt(f**2+g**2+h**2))
    phi7 = []
    for i in range(np.size(a)):
        if h[i] < 0:
            phi7 = np.append(phi7, 2*np.pi - np.arccos(g[i]/np.sqrt(g[i]**2 + h[i]**2)))
        else:
            phi7 = np.append(phi7, np.arccos(g[i]/np.sqrt(g[i]**2 + h[i]**2)))

    return [phi1, phi2, phi3, phi4, phi5, phi6, phi7]

def sample_hyperbolic_signal(d,n):

    x = 2*np.random.randn(d,n)
    t = np.sqrt(1 + np.sum(x**2,0))

    data = np.zeros((d+1,n))
    data[0:d,:] = x
    data[d,:] = t

    print('minkowsky inner-prod test : ', np.linalg.norm(1 + np.sum(data[0:d,:]**2,0) - data[d,:]**2))

    return data
    
def sample_smooth_hyperbolic_signal(d,n):

    x = 2*np.random.randn(d,n)
    
    xx = np.linspace(0,20*n,n)

    splx = []
    
    for i in range(d):
        splx = np.append(splx, CubicSpline(xx, x[i,:]))

    xxnew = np.linspace(0, 20*n, 20*n)

    data = np.zeros((d+1,np.size(xxnew)))

    for i in range(d):
        data[i,:] = splx[i](xxnew)

    t = np.sqrt(1 + np.sum(data**2,0))

    data[d,:] = t

    print('minkowsky inner-prod test : ', np.linalg.norm(1 + np.sum(data[0:d,:]**2,0) - data[d,:]**2))

    return data

def sample_smooth_hyperbolic_signal_h(d,n):

    x = 1*np.random.randn(d,n)
    
    xx = np.linspace(0,20*n,n)

    splx = []
    
    for i in range(d):
        splx = np.append(splx, CubicSpline(xx, x[i,:]))

    xxnew = np.linspace(0, 20*n, 20*n)

    data = np.zeros((d+1,np.size(xxnew)))

    for i in range(d):
        data[i,:] = splx[i](xxnew)

    data0 = data.copy()
    if d == 1:
        data[0,:] = np.sinh(data0[0,:])
        data[1,:] = np.cosh(data0[0,:])
    if d ==2 :
        data[0,:] = np.sinh(6/4*data0[0,:])*np.sin(3/4*data0[1,:])
        data[1,:] = np.sinh(6/4*data0[0,:])*np.cos(3/4*data0[1,:])
        data[2,:] = np.cosh(6/4*data0[0,:])


    print('minkowsky inner-prod test : ', np.linalg.norm(1 + np.sum(data[0:d,:]**2,0) - data[d,:]**2))

    return data


#############################################
#
# operators
#
#############################################

def L(x,r,N):

    U = np.zeros((N-1,3,3), dtype=complex)

    U[:,1,0] = x[0:N-1]
    U[:,0,1] = np.conjugate(x[0:N-1])

    U[:,2,0] = x[1:N]
    U[:,0,2] = np.conjugate(x[1:N])

    U[:,1,2] = r
    U[:,2,1] = np.conjugate(r)

    return U

def adjL(U,N):
	x = np.zeros(N, dtype = complex)

	x[0:N-1] = U[:,1,0] + np.matrix.conj(U[:,0,1])
	x[1:N] = x[1:N] + np.squeeze(U[:,2,0] + np.matrix.conj(U[:,0,2]))
	r = np.squeeze(U[:,1,2] + np.matrix.conj(U[:,2,1]))

	return [x,r]

def opL(w, x, y, z, l, f, g, h, N):

    U = np.zeros((N-1,6,6), dtype=complex)

    U[:,0,2] = w[0:N-1] - 1j*x[0:N-1]
    U[:,0,3] = -y[0:N-1] - 1j*z[0:N-1]

    U[:,2,0] = w[0:N-1] + 1j*x[0:N-1]
    U[:,3,0] = -y[0:N-1] + 1j*z[0:N-1]

    U[:,1,2] = y[0:N-1] - 1j*z[0:N-1]
    U[:,1,3] = w[0:N-1] + 1j*x[0:N-1]

    U[:,2,1] = y[0:N-1] + 1j*z[0:N-1]
    U[:,3,1] = w[0:N-1] - 1j*x[0:N-1]

    ###

    U[:,0,4] = w[1:N] - 1j*x[1:N]
    U[:,0,5] = -y[1:N] - 1j*z[1:N]

    U[:,4,0] = w[1:N] + 1j*x[1:N]
    U[:,5,0] = -y[1:N] + 1j*z[1:N]

    U[:,1,4] = y[1:N] - 1j*z[1:N]
    U[:,1,5] = w[1:N] + 1j*x[1:N]

    U[:,4,1] = y[1:N] + 1j*z[1:N]
    U[:,5,1] = w[1:N] - 1j*x[1:N]

    ###

    U[:,2,4] = l - 1j*f
    U[:,2,5] = -g - 1j*h
    
    U[:,4,2] = l + 1j*f
    U[:,5,2] = -g + 1j*h

    U[:,3,4] = g - 1j*h
    U[:,3,5] = l + 1j*f

    U[:,4,3] = g + 1j*h
    U[:,5,3] = l - 1j*f

    return U

def adjopL(U,N):
    
    w = np.zeros(N)
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    
    w[0:N-1] = np.real(U[:,2,0] + U[:,3,1] + U[:,0,2] + U[:,1,3])
    w[1:N] = w[1:N] + np.real(U[:,4,0] + U[:,5,1] + U[:,0,4] + U[:,1,5])

    x[0:N-1] = np.imag(U[:,2,0] + np.matrix.conj(U[:,3,1]) + np.matrix.conj(U[:,0,2]) + U[:,1,3])
    x[1:N] = x[1:N] + np.imag(U[:,4,0] + np.matrix.conj(U[:,5,1]) + np.matrix.conj(U[:,0,4]) + U[:,1,5])

    y[0:N-1] = np.real(U[:,2,1] - U[:,3,0] - U[:,0,3] + U[:,1,2])
    y[1:N] = y[1:N] + np.real(U[:,4,1] - U[:,5,0] - U[:,0,5] + U[:,1,4])

    z[0:N-1] = np.imag(U[:,2,1] + U[:,3,0] + np.matrix.conj(U[:,0,3]) + np.matrix.conj(U[:,1,2]))
    z[1:N] = z[1:N] + np.imag(U[:,4,1] + U[:,5,0] + np.matrix.conj(U[:,0,5]) + np.matrix.conj(U[:,1,4]))
    
    ''' where the complex conjugation, has to be shifted onto the operator L^* since 
            < Lq , U >_F = < q , L^*U >_R
        which is just definied be mulitplication with, since we are not on C^N but R^(4)^N includes q
    '''

    l = np.real(U[:,4,2] + U[:,5,3] + U[:,2,4] + U[:,3,5])
    f = np.imag(U[:,4,2] + np.conjugate(U[:,5,3]) + np.conjugate(U[:,2,4]) + U[:,3,5])
    g = np.real(U[:,4,3] - U[:,5,2] - U[:,2,5] + U[:,3,4])
    h = np.imag(U[:,4,3] + U[:,5,2] + np.conjugate(U[:,2,5]) + np.conjugate(U[:,3,4]))
    
    return [w, x, y, z, l, f, g, h]

def L_red(x,r,N,d):

    L = np.zeros((N-1,d+2,d+2))

    L[:,d,0:d] = x[:,0:N-1].T
    L[:,d+1,0:d] = x[:,1:N].T

    L[:,0:d,d] = x[:,0:N-1].T
    L[:,0:d,d+1] = x[:,1:N].T

    L[:,d,d+1] = r
    L[:,d+1,d] = r

    return L

def adjL_red(U,N,d):

    x = np.zeros((d,N))

    x[:,0:N-1] = 2*U[:,0:d,d].T

    x[:,1:N] = x[:,1:N] + 2*U[:,0:d,d+1].T

    r = 2*U[:,d,d+1]

    return [x,r]

def prox(U,N):

	for i in range(0,N-1):
		[D, V] = np.linalg.eigh(U[i,:,:])

		D = np.diag(np.minimum(np.real(D),0))

		U[i,:,:] = V@D@np.transpose(np.conjugate(V))

	return U

def D_1dim(n):

    D = np.zeros((n-1,n))

    for i in range(n-1):
        D[i,i] = 1 
        D[i,i+1] = -1

    return D

def ADMMprox(L, N):

	'''
	proximity operator for the real-valued model
	'''

	for i in range(0,N-1):
		
		[D, V] = np.linalg.eigh(L[i,:,:])
		
		D = np.diag(np.maximum(np.real(D),-1))

		L[i,:,:] = V@D@np.transpose(np.conjugate(V))

	return L

def ADMMproxC(L, N):

	'''
	proxity operator for the complex-valued model
	'''
		
	for i in range(0,N-1):
		
		[D, V] = np.linalg.eigh(L[i,:,:])
		
		D = np.diag(np.maximum(np.real(D),-1))

		L[i,:,:] = V@D@np.transpose(np.conjugate(V))

	return L

def proj_B1(x):

        d, n = np.shape(x)
        
        for i in range(n):
            if np.sum(x[:,i]**2) > 1:
                x[:,i] = x[:,i]/np.sqrt(np.sum(x[:,i]**2))
            
        return x

def opLHyper(x, v, f, ll):

    d, N = np.shape(x)
    l = d-1

    L = np.zeros((d+4, d+4, N-1))

    L[l+1,0:d,:] = x[:,0:N-1]
    L[l+2,0:d,:] = x[:,0:N-1]
    L[l+2,d-1,:] = -L[l+2,d-1,:]
    L[l+3,0:d,:] = x[:,1:N]
    L[l+4,0:d,:] = x[:,1:N]
    L[l+4,d-1,:] = -L[l+4,d-1,:]

    L[0:d,l+1,:] = x[:,0:N-1]
    L[0:d,l+2,:] = x[:,0:N-1]
    L[d-1,l+2,:] = -L[d-1,l+2,:]
    L[0:d,l+3,:] = x[:,1:N]
    L[0:d,l+4,:] = x[:,1:N]
    L[d-1,l+4,:] = -L[d-1,l+4,:]

    ####

    L[d, d, :] = v[0:N-1]
    L[d+1, d+1, :] = v[0:N-1]
    L[d+2, d+2, :] = v[1:N]
    L[d+3, d+3, :] = v[1:N]

    ####

    L[d+2, d,:] = f
    L[d+3, d+1,:] = f
    L[d, d+2,:] = f
    L[d+1, d+3,:] = f

    ####

    L[d+2, d+1,:] = ll
    L[d+3, d,:] = ll
    L[d+1, d+2,:] = ll
    L[d, d+3,:] = ll

    return L

def adjopLHyper(U):

    r,r,M = np.shape(U)
    #print(np.shape(U))

    N = M + 1
    d = r-4
    #print(d, N)

    x = np.zeros((d, N))
    v = np.zeros(N)
    f = np.zeros(M)
    l = np.zeros(M)

    x[:,0:N-1] += U[r-4,0:d,:] + U[r-3,0:d,:] + U[0:d,r-4,:] + U[0:d,r-3,:]
    x[d-1,0:N-1] += U[r-4,d-1,:] - U[r-3,d-1,:] + U[d-1,r-4,:] - U[d-1,r-3,:]
    x[:,1:N] += U[r-2,0:d,:] + U[r-1,0:d,:] + U[0:d,r-2,:] + U[0:d,r-1,:]
    x[d-1,1:N] += U[r-2,d-1,:] - U[r-1,d-1,:] + U[d-1,r-2,:] - U[d-1,r-1,:]

    v[0:N-1] = U[r-4,r-4,:] + U[r-3,r-3,:]
    v[1:N] += U[r-2,r-2,:] + U[r-1,r-1,:]

    f = U[r-2,r-4,:] + U[r-1,r-3,:] + U[r-4,r-2,:] + U[r-3,r-1,:]
    l = U[r-1,r-4,:] + U[r-2,r-3,:] + U[r-3,r-2,:] + U[r-4,r-1,:]

    return [x, v, f, l]

def proxHyper(W):

    r,r,M = np.shape(W)

    WW = np.zeros((r,r,M))
    for i in range(M):
        D, R = np.linalg.eig(W[:,:,i])
        DD = np.maximum(np.real(D), 0)
        WW[:,:,i] = R@np.diag(DD)@R.T
    return WW



#############################################
#
# solvers
#
#############################################

def PMM_S1(y, y0, lam, iter, tau, sph, eps):
    
    print('iteration \t| func-value \t| non-convex-cost \t| spherical-error')
    print('-----------------------------------------------------------------------')
    
    '''an equivalent algorithm is given by changing the f(x)+g(x) <- min  <> -f(x)-g(x) <- max, where:
		- the updates: 	adj_x = adj_x +/- y, adj_r = adj_r +/- lam
						x = x +/- tau*adj_x, r = r +/- tau*adj_r
						U = prox(U -/+ sigma*(L(x +/- tau*adj_x, r +/- tau*adj_r, N) + I), N)
		- the prioximanl mapping, since g -> -g, thus min(eigs, 0) -> max(eigs, 0)
	'''

    N = np.size(y)
	#x = y
    x = np.zeros(N, dtype=complex)
    r = x[0:N-1]*np.conjugate(x[1:N])
    U = np.zeros((N-1,3,3), dtype=complex)

    I = np.zeros((N-1,3,3), dtype=complex)
    I[:,0,0] = 1 + 0*1j
    I[:,1,1] = 1 + 0*1j
    I[:,2,2] = 1 + 0*1j

    #sigma = 1/tau/4
    sigma = 1/(tau*2*2)

    data = np.zeros(iter)
    datatime = np.zeros(iter, dtype=float)

    k = 0

    starttime = time.time()

    flaggg = 0
    flagggg = 0

    for j in range(0,iter):
        [adj_x, adj_r] = adjL(U,N)
        adj_x = adj_x - y
        adj_r = adj_r - lam

        if sph == 0:

            x = x - tau*adj_x
            r = r - tau*adj_r

            U = prox(U + sigma*(L(x - tau*adj_x, r - tau*adj_r, N) + I), N)

        if sph == 1:
            U = prox(U + sigma*(L((x - 2*tau*adj_x)/np.abs(x - 2*tau*adj_x), r - 2*tau*adj_r, N) + I), N)
            
            x = (x - tau*adj_x)/np.abs(x - tau*adj_x)
            r = r - tau*adj_r

        flag = 'unsphered'
        
        data[j] = np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(r))

        datatime[j] = time.time()

        if np.linalg.norm(adj_r) + np.linalg.norm(adj_x) < 10**(-eps) and flagggg == 0:
            print('iteration :', j, datatime[j] - starttime)
            flagggg = 1

        if np.mod(j,50)==0:

            if np.linalg.norm(1 - np.sum(np.sqrt(np.abs(x)))) < 1e-6:
                flag = 'sphered'

            k = k + 1

            print(j, '\t\t|', "%10.3e"%(data[j]) , '\t| ', "%10.3e"%(np.sum(1-np.cos(np.angle(x)-np.angle(y0)))+lam*np.sum(1-np.cos(np.angle(x[0:N-1])-np.angle(x[1:N])))), '\t\t|', "%10.3e"%(1 - np.mean(np.abs(x))))

    diff_x = x - y0
    x = x/np.abs(x)

    print('finale', '\t\t|', "%10.3e"%(np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(x[0:N-1]*np.conjugate(x[1:N])))) , '\t| ', "%10.3e"%(np.sum(1-np.cos(np.angle(x)-np.angle(y0)))+lam*np.sum(1-np.cos(np.angle(x[0:N-1])-np.angle(x[1:N])))), '\t\t| ' , flag)

    w = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(w+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        w = w + 1
    print(w, data[w], datatime[w] - starttime)

    for w in range(iter-1):
        if np.abs(data[w+1]-data[w])<10**(-9) and flaggg == 0:
            print(data[w], datatime[w] - starttime)
            flaggg = 1

    if sph==0:
        datei = open('data_S1_1Dgrid_pmm.txt','a')
        for w in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(data[w]))
            datei.write('\n')
        datei.write(str(np.sqrt(np.sum(diff_x**2))))
        datei.write('\n')
        datei.write(str(np.sum(np.abs(diff_x)**2)))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(np.real(x*np.conj(y0)))**2)))
        datei.write('\n')
        datei.close()

    if sph==1:
        datei = open('data_S1_1Dgrid_modpmm.txt','a')
        for w in range(np.size(data)):
            datei.write(str(data[w]))
            datei.write('\n')
        datei.write(str(np.linalg.norm(diff_x)))
        datei.write('\n')
        datei.write(str(np.sum(np.abs(diff_x)**2)))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(np.real(x*np.conj(y0))))))
        datei.write('\n')
        datei.close()

    return [x, data]

def PMM_S2_S3(Noise, Data, lam, iter, tau, rho, circ, eps):

    a = Noise[0,:]
    b = Noise[1,:]
    c = Noise[2,:]
    d = Noise[3,:]
    aa = Data[0,:]
    bb = Data[1,:]
    cc = Data[2,:]
    dd = Data[3,:]

    if Noise[0,0] == 0:
        d = 3
    else:
        d = 4

    N = np.size(a) 
    
    w = np.zeros(N)
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    l = np.zeros(N-1)
    f = np.zeros(N-1)
    g = np.zeros(N-1)
    h = np.zeros(N-1)

    U = np.zeros((N-1,6,6), dtype=complex)

    Id = np.zeros((N-1,6,6), dtype=complex)

    Id[:,0,0] = 1 + 0*1j
    Id[:,1,1] = 1 + 0*1j
    Id[:,2,2] = 1 + 0*1j
    Id[:,3,3] = 1 + 0*1j
    Id[:,4,4] = 1 + 0*1j
    Id[:,5,5] = 1 + 0*1j

    sigma = 1/(2*4*tau)

    data = np.zeros(iter, dtype=complex)
    datatime = np.zeros(iter, dtype=float)

    print('iteration \t| func-value \t| original-cost \t| spherical-error')
    print('------------------------------------------------------------------------')

    starttime = time.time()
    
    #print(np.sum(1 - np.cos(w-a)) + np.sum(1 - np.cos(x-b)) + np.sum(1 - np.cos(y-c)) + np.sum(1 - np.cos(z-d)) + lam*np.sum(1 - np.cos(w[0:N-1]-w[1:N])) + lam*np.sum(1 - np.cos(x[0:N-1]-x[1:N])) + lam*np.sum(1 - np.cos(y[0:N-1]-y[1:N])) + lam*np.sum(1 - np.cos(z[0:N-1]-z[1:N])))
    #print(np.sum(1 - w*a - x*b - y*c - z*d) + lam*np.sum(1 - w[0:N-1]*w[1:N] - x[0:N-1]*x[1:N] - y[0:N-1]*y[1:N] - z[0:N-1]*z[1:N]))

    for i in range(0,iter):
        
        [adj_w, adj_x, adj_y, adj_z, adj_l, adj_f, adj_g, adj_h] = adjopL(U, N)

        adj_w = adj_w - a
        adj_x = adj_x - b
        adj_y = adj_y - c
        adj_z = adj_z - d

        adj_l = adj_l - lam
        #adj_e = adj_e - lam
        #adj_f = adj_f - lam    #especially +/-0 since e,f,g not included in f(.) = <c,.>
        #adj_g = adj_g - lam

        if circ == 0:

            U = U + rho*(prox(U + sigma*(opL(w - 2*tau*adj_w, x - 2*tau*adj_x, y - 2*tau*adj_y, z - 2*tau*adj_z, l - 2*tau*adj_l, f - 2*tau*adj_f, g - 2*tau*adj_g, h - 2*tau*adj_h, N) + Id), N) - U)

            w = w - rho*tau*adj_w
            x = x - rho*tau*adj_x
            y = y - rho*tau*adj_y
            z = z - rho*tau*adj_z

            l = l - rho*tau*adj_l
            f = f - rho*tau*adj_f
            g = g - rho*tau*adj_g
            h = h - rho*tau*adj_h
        
        if circ == 1:

            ow = w - 2*rho*tau*adj_w
            ox = x - 2*rho*tau*adj_x
            oy = y - 2*rho*tau*adj_y
            oz = z - 2*rho*tau*adj_z

            onorm = np.sqrt(ow**2+ox**2+oy**2+oz**2)

            ow = ow/onorm
            ox = ox/onorm
            oy = oy/onorm
            oz = oz/onorm

            U = U + rho*(prox(U + sigma*(opL(ow, ox, oy, oz, l - 2*tau*adj_l, f - 2*tau*adj_f, g - 2*tau*adj_g, h - 2*tau*adj_h, N) + Id), N) - U)

            w = w - rho*tau*adj_w
            x = x - rho*tau*adj_x
            y = y - rho*tau*adj_y
            z = z - rho*tau*adj_z

            norm = np.sqrt(w**2+x**2+y**2+z**2)

            w = w/norm
            x = x/norm
            y = y/norm
            z = z/norm

            l = l - rho*tau*adj_l
            f = f - rho*tau*adj_f
            g = g - rho*tau*adj_g
            h = h - rho*tau*adj_h

        norm = w**2+x**2+y**2+z**2
        flag = 'rotation'

        data[i] = np.sum(1 - w*a - x*b - y*c - z*d) + lam*np.sum(1 - w[0:N-1]*w[1:N] - x[0:N-1]*x[1:N] - y[0:N-1]*y[1:N] - z[0:N-1]*z[1:N])

        datatime[i] = time.time()

        if np.mod(i,100) == 0:
            if np.linalg.norm(norm-np.ones(N))>np.exp(-6):
                flag = 'unsphered'

            print( i, '\t\t| ', "%10.3e"%(data[i]), '\t| ', "%10.3e"%(np.sum(1 - w*aa - x*bb - y*cc - z*dd) + lam*np.sum(1 - w[0:N-1]*w[1:N] - x[0:N-1]*x[1:N] - y[0:N-1]*y[1:N] - z[0:N-1]*z[1:N])), '\t\t| ', "%10.3e"%(1 - np.abs(np.mean(norm))))
            

    norm = np.sqrt(w**2+x**2+y**2+z**2)

    diff_W = w - aa
    diff_X = x - bb
    diff_Y = y - cc
    diff_Z = z - dd

    if circ == 0:
        datei = open('data_S{0}_1Dgrid_pmm.txt'.format(d-1),'a')
        for r in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(np.real(data[r])))
            datei.write('\n')
        datei.write(str(np.sqrt(np.sum(diff_W**2+diff_X**2+diff_Y**2+diff_Z**2))))
        datei.write('\n')
        datei.write(str(np.sum(diff_W**2+diff_X**2+diff_Y**2+diff_Z**2)))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(w*aa + x*bb + y*cc + z*dd)**2)))
        datei.write('\n')
        datei.close()

    if circ == 1:
        datei = open('data_S{0}_1Dgrid_modpmm.txt'.format(d-1),'a')
        for r in range(np.size(data)):
            #map(data[w], datei.readlines())
            datei.write(str(np.real(data[r])))
            datei.write('\n')
        datei.write(str(np.sqrt(np.sum(diff_W**2+diff_X**2+diff_Y**2+diff_Z**2))))
        datei.write('\n')
        datei.write(str(np.sum(np.arccos(w*aa + x*bb + y*cc + z*dd)**2)))
        datei.write('\n')
        datei.close()
    
    r = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(r+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        r = r + 1
    print(r, data[r], datatime[r] - starttime)

    return [w, x, y, z, data]

	

def baseline(y, y0, lam, iter, circ):

    d, N = np.shape(y) 
    f = d-1

    nx = np.zeros((d, N))
    x = np.zeros((d, N))

    D = np.transpose(D_1dim(N))@D_1dim(N)

    data = np.zeros(iter)
    datatime = np.zeros(iter)
    datei = open('data_S{0}_1Dgrid_ppa.txt'.format(f),'a')

    starttime = time.time()

    print('iter. \t\t| func-value \t| non-convex-cost \t| solution is')
    print('--------------------------------------------------------------------------')

    for s in range(iter):

        for j in range(d):
            grad = (x[j,:] - y[j,:]) + lam*D@x[j,:].T

            x[j,:] = x[j,:] - grad/(4*lam + 1)

        norm = np.zeros(N)
        for j in range(N):
            norm[j] = np.sum(x[:,j]**2,0) 

        norm = np.sqrt(norm) 
        flag = 'sphered'

        for j in range(N):
            if norm[j] < 1:
                if circ == 1:
                    x[:,j] = x[:,j]/norm[j]
                flag = 'unsphered'
        
        datatime[s] = time.time()

        data[s] = np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - np.sum(x[:,0:N-1]*x[:,1:N],0)) 

        datei.write(str(np.real(data[s])))
        datei.write('\n')

        if np.mod(s,100)==0:
                print(s, '\t\t|', "%10.3e"%(data[s]), '\t|', "%10.3e"%(np.sum(1-np.cos(np.angle(x[0,:] + 1j*x[1,:])-np.angle(y[0,:] + 1j*y[1,:])))+lam*np.sum(1-np.cos(np.angle(x[0,0:N-1]+1j*x[1,0:N-1])-np.angle(x[0,1:N]+1j*x[1,1:N])))), '\t\t|', flag)

    norm = np.zeros(N)
    for j in range(N):
        norm[j] = np.sqrt(np.sum(x[:,j]**2,0)) 
    for j in range(N):
        nx[:,j] = x[:,j]/norm[j]

    print('finale','\t\t|', "%10.3e"%(np.sum(1 - np.sum(nx*y,0)) + lam*np.sum(1 - np.sum(nx[:,0:N-1]*nx[:,1:N],0))) , '\t|', "%10.3e"%(np.sum(1-np.cos(np.angle(nx[0,:] + 1j*nx[1,:])-np.angle(y[0,:] + 1j*y[1,:])))+lam*np.sum(1-np.cos(np.angle(nx[0,0:N-1]+1j*nx[1,0:N-1])-np.angle(nx[0,1:N]+1j*nx[1,1:N])))), '\t\t|', flag)

    diff_X = x - y0
    
    datei.write(str(np.sqrt(np.sum(diff_X**2))))
    datei.write('\n')
    datei.write(str(np.sum(diff_X**2)))
    datei.write('\n')
    datei.write(str(np.sum(np.arccos(np.sum(nx*y0,0))**2)))
    datei.write('\n')
    datei.close()

    w = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(w+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), 3) + 0.001 or data[ww] < truncate(np.real(data[iter-1]), 3) - 0.001:
                flagg = 0
        w = w + 1
    print(w, data[w], datatime[w] - starttime)

    return [nx, data]

def ADMM_S1(y, y0, lam, rho, iter, eps):

    ''' 
    rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
    the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
    Hence, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''
    w = y0

    N = np.size(y)
    x = np.zeros(N, dtype='cdouble')
    r = np.zeros(N-1)
    s = np.zeros(N-1)
    #x = y
    #r = y[0:N-1]*np.conjugate(y[1:N])

    Z = np.zeros((N-1,3,3), dtype='cdouble')
    U = np.zeros((N-1,3,3), dtype='cdouble')

    data = np.zeros(iter)
    datatime = np.zeros(iter)

    flaggg = 0
    flagggg = 0

    print('iteration \t| func-value \t| maginal-cost \t| non-convex-cost \t| spherical-error')
    print('------------------------------------------------------------------------------------------')

    starttime = time.time()
    
    for i in range(iter):

        [adjZx, adjZr] = adjL(Z, N)
        [adjUx, adjUr] = adjL(U, N)

        # s <- argmin_s f(s) + rho/2*||Ls - u + z||^2 ------- first ADMM-step
        x[0] = 1/2*(adjUx[0] - adjZx[0] + 1/rho*y[0])
        x[1:N-1] = 1/4*(adjUx[1:N-1] - adjZx[1:N-1] + 1/rho*y[1:N-1])
        x[N-1] = 1/2*(adjUx[N-1] - adjZx[N-1] + 1/rho*y[N-1])
        r = 1/2*(np.real(adjUr) - np.real(adjZr) + 1/rho*lam)
        s = 1/2*(np.imag(adjUr) - np.imag(adjZr))

        # U <- argmin (.) = prox_{hpsd + I => 0}(.) ------- second ADMM-step
        temp = L(x, r + 1j*s, N)
        Utemp = U
        U = ADMMproxC(temp.copy() + Z, N)

        # Z <- Z + Ls - U ------- third ADMM-step // update
        Z += temp - U

        flag = 'unsphered'

        data[i] =  np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - r)

        datatime[i] = time.time()

        if np.linalg.norm(temp - U) < 10**(-eps) and np.linalg.norm(Utemp - U) < 10**(-eps) and flagggg == 0:
            print('iteration :', i, datatime[i] - starttime)
            flagggg = 1

        if np.mod(i,100) == 0:
            if np.linalg.norm(1 - np.sum(np.sqrt(np.abs(x)))) < 1e-6:
                flag = 'sphered'
            
            print(i, '\t\t|', "%10.3e"% data[i] , '\t|', "%10.3e"% (np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(x[0:N-1]*np.conjugate(x[1:N])))), '\t|', "%10.3e"% (np.sum(1-np.cos(np.angle(x)-np.angle(y0)))+lam*np.sum(1-np.cos(np.angle(x[0:N-1])-np.angle(x[1:N])))), '\t\t|', "%10.3e"% (1 - np.mean(np.abs(x))))

    print('finale','\t\t|', "%10.3e"% (np.sum(1 - np.real(x*np.conjugate(y))) + lam*np.sum(1 - np.real(x[0:N-1]*np.conjugate(x[1:N])))) , '\t|', "%10.3e"% (np.sum(1-np.cos(np.angle(x)-np.angle(y)))+lam*np.sum(1-np.cos(np.angle(x[0:N-1])-np.angle(x[1:N])))), '\t\t\t\t|', flag)

    w = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(w+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        w = w + 1
    print(w, data[w], datatime[w] - starttime)

    for w in range(iter-1):
        if np.abs(data[w+1]-data[w])<10**(-9) and flaggg == 0:
            print(data[w], datatime[w] - starttime)
            flaggg = 1
    
    diff_x = x - y0
    nx = x/np.abs(x)

    datei = open('data_S1_1Dgrid_admm.txt','a')
    for w in range(np.size(data)):
		#map(data[w], datei.readlines())
        datei.write(str(data[w]))
        datei.write('\n')
    datei.write(str(np.linalg.norm(diff_x)))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(diff_x)**2)))
    datei.write('\n')
    datei.write(str(np.sum(np.arccos(np.real(x*np.conjugate(y0)))**2)))
    datei.write('\n')
    datei.close()
    
    return [x, data]

def ADMM_red(y, y0, lam, rho, iter, eps):

    ''' 
    rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
    the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
    Hence, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    d, N = np.shape(y)
    x = np.zeros((d,N), dtype='float64')
    r = np.zeros(N-1, dtype='float64')
    #x = y
    #r = y[0:N-1]*np.conjugate(y[1:N])

    Z = np.zeros((N-1,d+2,d+2), dtype='float64')
    U = np.zeros((N-1,d+2,d+2), dtype='float64')

    data = np.zeros(iter, dtype='float64')
    datatime = np.zeros(iter, dtype='float64')

    flaggg = 0
    flagggg = 0

    print('iteration \t| func-value \t| marginal-cost \t| sherical-error')
    print('--------------------------------------------------------------------------')

    starttime = time.time()
    
    for i in range(iter):

        [adjZx, adjZr] = adjL_red(Z, N, d)
        [adjUx, adjUr] = adjL_red(U, N, d)

        #s <- argmin_s f(s) + rho/2*||Ls - u + z||^2  ------- first ADMM-step
        x[:,0] = 1/2*(adjUx[:,0] - adjZx[:,0] + 1/rho*y[:,0])
        x[:,1:N-1] = 1/4*(adjUx[:,1:N-1] - adjZx[:,1:N-1] + 1/rho*y[:,1:N-1])
        x[:,N-1] = 1/2*(adjUx[:,N-1] - adjZx[:,N-1] + 1/rho*y[:,N-1])
        r = 1/2*(adjUr - adjZr + 1/rho*lam)

        #U <- argmin (.) = prox_{hpsd + I => 0}(.)  ------- second ADMM-step
        temp = L_red(x, r, N, d)
        Utemp = U
        U = ADMMprox(temp.copy() + Z, N)

        #Z <- Z + Ls - U  ------- third ADMM-step // update
        Z += temp - U

        flag = 'unsphered'

        data[i] =  np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - r)

        datatime[i] = time.time()

        if np.linalg.norm(temp - U) < 10**(-eps) and np.linalg.norm(Utemp - U) < 10**(-eps) and flagggg == 0:
            print('iteration :', i, datatime[i] - starttime)
            flagggg = 1

        if np.mod(i,50) == 0:
            if np.linalg.norm(1 - np.sqrt(np.sum(x**2,0))) < 1e-6:
                flag = 'sphered'
            
            print( i , '\t\t|', "%10.3e"% (data[i]) , '\t|', "%10.3e"% (np.sum(1-np.cos(np.angle(x[0,:] + 1j*x[1,:])-np.angle(y[0,:] + 1j*y[1,:])))+lam*np.sum(1-np.cos(np.angle(x[0,0:N-1]+1j*x[1,0:N-1])-np.angle(x[0,1:N]+1j*x[1,1:N])))) , '\t\t|', "%10.3e"% (1 - np.mean(np.sum(x**2,0))))

    w = 0
    flagg = 0
    while flagg == 0:
        flagg = 1
        for ww in range(w+1,iter):
            if data[ww] > truncate(np.real(data[iter-1]), eps) + 10**(-eps) or data[ww] < truncate(np.real(data[iter-1]), eps) - 10**(-eps):
                flagg = 0
        w = w + 1
    print(w, data[w], datatime[w] - starttime)

    for w in range(iter-1):
        if np.abs(data[w+1]-data[w])<10**(-9) and flaggg == 0:
            print(data[w], datatime[w] - starttime)
            flaggg = 1

    diff_x = x - y0
    #nx = x/np.abs(x)

    datei = open('data_S{0}_1Dgrid_admm_red.txt'.format(d-1),'a')
    for w in range(np.size(data)):
        datei.write(str(data[w]))
        datei.write('\n')
    datei.write(str(np.sqrt(np.sum(diff_x**2))))
    datei.write('\n')
    datei.write(str(np.sum(diff_x**2)))
    datei.write('\n')
    datei.write(str(np.sum(np.arccos(np.sum(x*y0,0))**2)))
    datei.write('\n')
    datei.close()

    print('finale','\t\t|', "%10.3e"% (np.sum(1 - np.sum(x*y,0)) + lam*np.sum(1 - np.sum(x[:,0:N-1]*x[:,1:N],0))) , '\t|', "%10.3e"% (np.sum(1-np.cos(np.angle(x[0,:] + 1j*x[1,:])-np.angle(y[0,:] + 1j*y[1,:])))+lam*np.sum(1-np.cos(np.angle(x[0,0:N-1]+1j*x[1,0:N-1])-np.angle(x[0,1:N]+1j*x[1,1:N])))), '\t\t|', flag)

    return [x, data]

def ADMM_red_torus_1D(y, y_0, lam, rho, iter):

    d, N = np.shape(y)
    r = int(d/2)
    M = N-1

    x = np.zeros((d, N))
    l = np.zeros((r, M))
    U = np.zeros((r,M,4,4))
    Z = np.zeros((r,M,4,4))

    print('iteration \t| func-value \t| torus-error')
    print('---------------------------------------------------')

    for i in range(iter):

        for k in range(r):
            [adjx, adjl] = adjL_red(U[k,:,:,:] - Z[k,:,:,:], N, 2)

            #s <- argmin_s f(s) + rho/2*||Ls - u + z||^2  ------- first ADMM-step
            x[2*k:2*k+2,0] = 1/2*(adjx[:,0] + 1/rho*y[2*k:2*k+2,0])
            x[2*k:2*k+2,1:N-1] = 1/4*(adjx[:,1:N-1] + 1/rho*y[2*k:2*k+2,1:N-1])
            x[2*k:2*k+2,N-1] = 1/2*(adjx[:,N-1] + 1/rho*y[2*k:2*k+2,N-1])
            l[k,:] = 1/2*(adjl + 1/rho*lam)

            #U <- argmin (.) = prox_{hpsd + I => 0}(.)  ------- second ADMM-step
            temp = L_red(x[2*k:2*k+2,:], l[k,:], N, 2)
            #Utemp = U
            U[k, :,:,:] = ADMMprox(temp.copy() + Z[k, :,:,:], N)

            #Z <- Z + Ls - U  ------- third ADMM-step // update
            Z[k, :,:,:] += temp - U[k,:,:,:]
        
        norm = np.zeros((r, N))
        for w in range(r):
            norm[w,:] = np.sum(x[2*w:2*w+2,:]**2, 0)
        
        if np.mod(i, 100) == 0:
            print(i ,'\t\t|', "%10.2e"% (np.sum(-np.sum(x*y, 0)) - lam*np.sum(l)), '\t|', "%10.2e"% np.linalg.norm(1 - norm))
    
    return [x, l]

def ADMM_TV_BOX(y, y0, mu, rho, iter):

        d, n = np.shape(y)

        x = np.zeros((d, n))
        u = np.zeros((d, n))
        z = np.zeros((d, n))

        print('iteration \t| funv-value \t\t| RMSE \t\t| error \t| spherical-error')
        print('----------------------------------------------------------------------------------')

        x11 = np.random.randn(d, n)

        j = 0

        flag = np.linalg.norm(np.sqrt(np.sum(x**2,0)) - 1)

        while np.linalg.norm(x11 - x) > 1e-6 or flag > 1e-6 and j < 30000:
            x11 = np.copy(x)

            # argmin_x  -<x,y> + mu|Dx|_1 + iota(u) + rho/2||x - u + z||_2
            for l in range(d):
                x[l,:] = condat_tv.tv_denoise(u[l,:] - z[l,:] + y[l,:]/rho, mu/rho)

            flag = np.linalg.norm(np.sqrt(np.sum(x**2,0)) - 1)

            # proj_B(1)
            u = proj_B1(x + z)

            # update
            z = z + x - u

            if np.mod(j,100) == 0:
                print(j, ' \t\t| ',np.linalg.norm(x - y)**2 + mu*np.sum(np.abs(x[0,0:d-1] - x[0,1:d])) + mu*np.sum(np.abs(x[1,0:d-1] - x[1,1:d])) + mu*np.sum(np.abs(x[2,0:d-1] - x[2,1:d])) , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2/np.shape(x)[1]), ' \t| ' , "%10.2e"% (np.linalg.norm(x11 - x)), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.sqrt(np.sum(x**2,0)))))

            j = j+1

        print('finale', ' \t\t| ',np.linalg.norm(x - y)**2 + mu*np.sum(np.abs(x[0,0:d-1] - x[0,1:d])) + mu*np.sum(np.abs(x[1,0:d-1] - x[1,1:d])) + mu*np.sum(np.abs(x[2,0:d-1] - x[2,1:d])) , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2/np.shape(x)[1]), ' \t| ' , "%10.2e"% (np.linalg.norm(x11 - x)), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.sqrt(np.sum(x**2,0)))))

        
        datei = open('data_S{0}_1Dgrid_box.txt'.format(d-1),'a')
        datei.write(str(np.linalg.norm(x - y0)**2))
        datei.write('\n')
        datei.write(str(np.sum(np.abs(x[0,:] - y0[0,:]) + np.abs(x[1,:] - y0[1,:]) + np.abs(x[2,:] - y0[2,:]))))
        datei.write('\n')
        datei.close()

        return [x, z]

def ADMM_red_hyper(y, y_0, lam, rho, iter):

    d, N = np.shape(y)

    U = np.zeros((d+4,d+4,N-1))
    Z = np.zeros((d+4,d+4,N-1))

    x = np.zeros((d,N))
    v = np.zeros(N)
    l = np.zeros(N-1)
    f = np.zeros(N-1)

    E = np.zeros((d+4,d+4,N-1))
    for i in range(d):
        E[i,i,:] = np.ones(N-1)
    E[d+1,d,:] = -np.ones(N-1)
    E[d,d+1,:] = -np.ones(N-1)
    E[d+3,d+2,:] = -np.ones(N-1)
    E[d+2,d+3,:] = -np.ones(N-1)

    x1 = np.random.randn(d,N)
    i = 0 

    print('iteration \t| func-value \t| mikwosky-error \t| error')
    print('--------------------------------------------------------------------------')

    #for i in range(iter):
    while np.linalg.norm(x1 - x) > 1e-4 and np.linalg.norm(1 + np.sum(x[0:d-1,:]**2,0) - x[d-1,:]**2) > 1e-4 and i < iter:

        adj_x, adj_v, adj_f, adj_l = adjopLHyper(U - Z)
        #adj_xz, adj_vz, adj_fz, adj_lz = adjopL(Z)

        #print(np.shape(adj_xu), np.shape(adj_vu), np.shape(adj_fu), np.shape(adj_lu))

        x1 = x.copy()

        x[:,0] = 1/4*(adj_x[:,0] + y[:,0]/rho)
        x[:,1:N-1] = 1/8*(adj_x[:,1:N-1] + y[:,1:N-1]/rho)
        x[:,N-1] = 1/4*(adj_x[:,N-1] + y[:,N-1]/rho)

        f = 1/4*(adj_f + lam/rho)

        v[0] = 1/2*(adj_v[0] - 1/2/rho - lam/2/rho)
        v[1:N-1] = 1/4*(adj_v[1:N-1] - 1/2/rho - lam/rho)
        v[N-1] = 1/2*(adj_v[N-1] - 1/2/rho - lam/2/rho)

        l = 1/4*(adj_l)

        ### --------------

        R = opLHyper(x, v, f, l) + Z + E
        #U = ADMMprox(opL(x, v, f, l), Z)
        U = proxHyper(R.copy()) - E

        ### --------------

        Z = Z + opLHyper(x, v, f, l) - U
        #Z = R.copy() - U

        if np.mod(i,100) == 0:
            print(i, '\t\t|', "%10.2e"% (np.sum(1/2*(np.sum(y**2,0) + np.sum(x**2,0) - 2*np.sum(x*y,0)) + lam/2*(np.sum(v[0:N-1]) + np.sum(v[1:N]) - 2*np.sum(f)))), '\t|', "%10.2e"% np.linalg.norm(1 + np.sum(x[0:d-1,:]**2,0) - x[d-1,:]**2), '\t\t|', "%10.2e"% np.linalg.norm(x1 - x))

        i += 1
        
    return [x, v, f, l]

#############################################
#
# plots
#
#############################################

def modline(x,y):
    n = x.size
    modx = np.copy(x)
    mody = np.copy(y)
    id = 0
    for k in range(n-1):
        id += 1
        if np.abs(y[k] - y[k+1]) > np.pi:
            if y[k] > y[k+1]:
                t = x[k] - (np.pi - y[k]) / (2 * np.pi + y[k+1] - y[k]) * (x[k] - x[k+1])
                modx = np.insert(modx, [id, id, id], [t, t, t])
                mody = np.insert(mody, [id, id, id], [np.pi, np.nan, -np.pi])
            else:
                t = x[k] - (-np.pi - y[k]) / (y[k+1] - 2 * np.pi - y[k]) * (x[k] - x[k+1])
                modx = np.insert(modx, [id, id, id], [t, t, t])
                mody = np.insert(mody, [id, id, id], [-np.pi, np.nan, np.pi])
            id += 3
    return(modx, mody)

def modline_S7(x,y):
    n = x.size
    modx = np.copy(x)
    mody = np.copy(y)
    id = 0
    for k in range(n-1):
        id += 1
        if np.abs(y[k] - y[k+1]) > np.pi:
            if y[k] > y[k+1]:
                t = x[k] - (2*np.pi - y[k]) / (2 * np.pi + y[k+1] - y[k]) * (x[k] - x[k+1])
                modx = np.insert(modx, [id, id, id], [t, t, t])
                mody = np.insert(mody, [id, id, id], [2*np.pi, np.nan, 0])
            else:
                t = x[k] - ( - y[k]) / (y[k+1] - 2 * np.pi - y[k]) * (x[k] - x[k+1])
                modx = np.insert(modx, [id, id, id], [t, t, t])
                mody = np.insert(mody, [id, id, id], [0, np.nan, 2*np.pi])
            id += 3
    return(modx, mody)

def hackline(x,y):
    n = x.size
    modx = np.copy(x)
    mody = np.copy(y)
    id = 0
    for k in range(n-1):
        id += 1
        if np.abs(y[k] - y[k+1]) > np.pi:
            if y[k] > y[k+1]:
                t = x[k] - (2*np.pi - y[k]) / (2 * np.pi + y[k+1] - y[k]) * (x[k] - x[k+1])
                modx = np.insert(modx, [id, id, id], [t, t, t])
                mody = np.insert(mody, [id, id, id], [2*np.pi, np.nan, 0])
            else:
                t = x[k] - (- y[k]) / (y[k+1] - 2 * np.pi - y[k]) * (x[k] - x[k+1])
                modx = np.insert(modx, [id, id, id], [t, t, t])
                mody = np.insert(mody, [id, id, id], [0, np.nan, 2*np.pi])
            id += 3
    return(modx, mody)

def plotS1(Noise, Data, q):

    N = np.size(q[0,:])
    x = np.linspace(0,N-1,N)

    (modx, mody) = modline(x,np.angle(Noise[0,:] + 1j*Noise[1,:]))
    #(modx, mody) = modline(x,Noise)
    (modx0, mody0) = modline(x,np.angle(Data[0,:] + 1j*Data[1,:]))
    #(modx0, mody0) = modline(x,Data)
    (modq0, modq) = modline(x,np.angle(q[0,:] + 1j*q[1,:]))
    #(modq0, modq) = modline(x,q)
    fig = plt.figure(0,figsize=(15,1*3), dpi = 2*100)
    ax = fig.add_subplot()
    p1, = ax.plot(modx, mody,'k', linewidth=0.5) 
    p0, = ax.plot(modx0, mody0,'b') 
    p2, = ax.plot(modq0, modq,'r') 
    ax.margins(y=0.) 

def plotS2(Noise, Data, q):

    T0, P0 = angle_S2(Data[0,:], Data[1,:], Data[2,:]) 
    T, P = angle_S2(Noise[0,:], Noise[1,:], Noise[2,:]) 
    T1, P1 = angle_S2(q[0,:], q[1,:], q[2,:])
    
    fig = plt.figure(0,figsize=(15,2*3), dpi=2*100)
    plt.rc('font', size=7.5) 

    x = np.linspace(0,np.size(T)-1,np.size(T))

    (modx, mody) = modline(x,T)
    #(modx, mody) = modline(x,Noise)
    (modx0, mody0) = modline(x,T0)
    #(modx0, mody0) = modline(x,Data)
    (modq0, modq) = modline(x,T1)
    #(modq0, modq) = modline(x,q)
    ax = fig.add_subplot(2, 1, 1)
    p1, = ax.plot(modx, mody,'k', linewidth=0.5) 
    p0, = ax.plot(modx0, mody0,'b') 
    p2, = ax.plot(modq0, modq,'r') 
    ax.legend([p2, p1, p0], ['ADMMred', 'Noise', 'Ground truth'])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax1 = fig.add_subplot(2, 1, 2)
    p0, = ax1.plot(P0, 'b') 
    p2, = ax1.plot(P,'k', linewidth=0.5)
    p1, = ax1.plot(P1,'r') 
    ax1.legend([p1, p0, p2], ['ADMMred', 'Ground truth','Noise'])
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

def plotSO3(Noise, Data, q):

    [a, p, t] = angle_SO3(Noise)
    [a0, p0, t0] = angle_SO3(Data)
    [a1, p1, t1] = angle_SO3(q)
     
    plt.rc('font', size=7.5) 

    fig = plt.figure(figsize=(15,3*4), dpi = 4*100)

    ax = fig.add_subplot(4, 1, 1)
    ax.plot(a0, 'b') 
    s2, = ax.plot(a1,'r') 
    ax.plot(a,'k', linewidth=0.5) 
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax1 = fig.add_subplot(4, 1, 2)
    ax1.plot(p0, 'b') 
    s2, = ax1.plot(p1,'r') 
    ax1.plot(p,'k', linewidth=0.5) 
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    ax2 = fig.add_subplot(4, 1, 3)
    ax2.plot(t0, 'b') 
    s2, = ax2.plot(t1,'r') 
    ax2.plot(t,'k', linewidth=0.5) 
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

def plotSO3_camera(Noise, Data, qq):

    k = 50
    l = 50

    k = 10
    l = 300

    fig = plt.figure(figsize=(3,10), dpi=300)

    for i in range(15):
        cam2world = pt.transform_from_pq([-0.75, -1.6+i/4.5, 0, Data[0,k*i+l], Data[1,k*i+l], Data[2,k*i+l], Data[3,k*i+l]])
        
        # default parameters of a camera in Blender
        sensor_size = np.array([0.01, 0.01])
        intrinsic_matrix = np.array([
            [0.05, 0, sensor_size[0] / 2.0],
            [0, 0.05, sensor_size[1] / 2.0],
            [0, 0, 1]
        ])
        virtual_image_distance = 0.4

        ax = pt.plot_transform(A2B=cam2world, s=0.15)
        pc.plot_camera( 
            ax, cam2world=cam2world, 
            M=intrinsic_matrix, sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance, alpha=0.2)

        cam2world = pt.transform_from_pq([0, -1.6+i/4.5, 0, Noise[0,k*i+l], Noise[1,k*i+l], Noise[2,k*i+l], Noise[3,k*i+l]])
        
        # default parameters of a camera in Blender
        sensor_size = np.array([0.01, 0.01])
        intrinsic_matrix = np.array([
            [0.05, 0, sensor_size[0] / 2.0],
            [0, 0.05, sensor_size[1] / 2.0],
            [0, 0, 1]
        ])
        virtual_image_distance = 0.4


        ax = pt.plot_transform(A2B=cam2world, s=0.15)
        pc.plot_camera( 
            ax, cam2world=cam2world, 
            M=intrinsic_matrix, sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance, alpha=0.2)
        
        cam2world = pt.transform_from_pq([0.77, -1.6+i/4.5, 0, qq[0,k*i+l], qq[1,k*i+l], qq[2,k*i+l], qq[3,k*i+l]])

        # default parameters of a camera in Blender
        sensor_size = np.array([0.01, 0.01])
        intrinsic_matrix = np.array([
            [0.05, 0, sensor_size[0] / 2.0],
            [0, 0.05, sensor_size[1] / 2.0],
            [0, 0, 1]
        ])
        virtual_image_distance = 0.4

        ax = pt.plot_transform(A2B=cam2world, s=0.15)
        pc.plot_camera( 
            ax, cam2world=cam2world, 
            M=intrinsic_matrix, sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance, alpha=0.2)
    ax.view_init(azim=0, elev=90)
    ax.set_box_aspect((1,1,1.1))
    #ax.view_init(azim=120, elev=20)
    plt.axis('off')


def plotS7(Noise, Data, q):
    
    fig = plt.figure(dpi=400)

    ax = fig.add_subplot(projection='3d')

    xx = np.linspace(0, 999, 1000)

    p01, p02, p03, p04, p05, p06, p07 = angle_S7(Data)
    p1, p2, p3, p4, p5, p6, p7 = angle_S7(Noise)
    p11, p22, p33, p44, p55, p66, p77 = angle_S7(q)

    (modx, mody) = modline_S7(xx,p07)
    (modx0, mody0) = modline_S7(xx,p7)
    (modxx, modyy) = modline_S7(xx,p77)

    ax.plot(modx, mody, zs=6, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(modx0, mody0, zs=6, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(modxx, modyy, zs=6, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.plot(xx, p06, zs=5, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(xx, p6, zs=5, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(xx, p66, zs=5, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.plot(xx, p05, zs=4, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(xx, p5, zs=4, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(xx, p55, zs=4, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.plot(xx, p04, zs=3, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(xx, p4, zs=3, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(xx, p44, zs=3, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.plot(xx, p03, zs=2, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(xx, p3, zs=2, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(xx, p33, zs=2, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.plot(xx, p02, zs=1, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(xx, p2, zs=1, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(xx, p22, zs=1, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.plot(xx, p01, zs=0, zdir='y', alpha=1, color='b', linewidth=0.5)
    ax.plot(xx, p1, zs=0, zdir='y', alpha=0.5, color='k', linewidth=0.2)
    ax.plot(xx, p11, zs=0, zdir='y', alpha=1, color='r', linewidth=0.5)

    ax.set_zlim(0, 2*np.pi)
    ax.set_zticks(np.linspace(0,2*np.pi,3))
    ax.set_yticks([])
    ax.margins(y=0.) 
    ax.set_box_aspect((2.5,1.5,1))

    ax.view_init(elev=35., azim=-45, roll=0.2)

def plotTorus(Noise, Data, sol_x):

    from mpl_toolkits.mplot3d import Axes3D
    #import matplotlib.pyplot as plt
    #import numpy as np
    from itertools import product, combinations


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection = '3d')
    ax.set_aspect("auto")

    # draw torus
    r = 1
    R = 2
    u, v = np.mgrid[0:2*np.pi:40j, 0:2*np.pi:20j]
    x = (R + r*np.cos(u))*np.cos(v)
    y = (R + r*np.cos(u))*np.sin(v)
    z = r*np.sin(u)
    ax.plot_wireframe(x, y, z, color="grey", alpha= 0.4)

    # draw signal 
    u = np.angle(Data[0,:] + 1j*Data[1,:])
    v = np.angle(Data[2,:] + 1j*Data[3,:])
    x = (R + r*np.cos(u))*np.cos(v)
    y = (R + r*np.cos(u))*np.sin(v)
    z = r*np.sin(u)
    ax.plot(x, y, z, color="blue", alpha= 0.75)

    u = np.angle(Noise[0,:] + 1j*Noise[1,:])
    v = np.angle(Noise[2,:] + 1j*Noise[3,:])
    x = (R + r*np.cos(u))*np.cos(v)
    y = (R + r*np.cos(u))*np.sin(v)
    z = r*np.sin(u)
    ax.scatter(x, y, z, color="k", alpha= 0.25)

    u = np.angle(sol_x[0,:] + 1j*sol_x[1,:])
    v = np.angle(sol_x[2,:] + 1j*sol_x[3,:])
    x = (R + r*np.cos(u))*np.cos(v)
    y = (R + r*np.cos(u))*np.sin(v)
    z = r*np.sin(u)
    ax.plot(x, y, z, color="red", alpha= 0.75)


def plot_hyper1(Noise, Data, sol_x):

    fig = plt.figure(figsize=(15,3))
    plt.plot(np.arcsinh(Noise[0,:]), linewidth=0.75, color='black')
    #plt.plot(np.arccosh(noise_smooth_signal[1,:]), linewidth=0.75, color='black')
    plt.plot(np.arcsinh(Data[0,:]), color='blue')
    plt.plot(np.arcsinh(sol_x[0,:]), color='red')
    for pos in ['right', 'top']:
            plt.gca().spines[pos].set_visible(False)
    #plt.savefig('1-hyperbolic_data_denoise_h_presentation_sig=0.6.pdf', dpi=400)

def plot_hyper2(Noise, Data, sol_x):

    from mpl_toolkits.mplot3d import Axes3D
    from itertools import product, combinations


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection = '3d')
    ax.set_aspect("auto")

    # draw torus
    r = 1
    R = 2
    k = 3
    u, v = np.mgrid[-2*k:2*k:40j, -2*k:2*k:20j]
    x = u
    y = v
    z = np.sqrt(1 + x**2 + y**2) 
    ax.plot_wireframe(x, y, z, color="grey", alpha= 0.4)

    # draw signal 
    ax.plot(Data[0,:], Data[1,:], Data[2,:], color="blue", alpha= 1)

    # draw noise 
    ax.scatter(Noise[0,:], Noise[1,:], Noise[2,:], color="k", alpha= 0.25)

    # draw sol 
    ax.plot(sol_x[0,:], sol_x[1,:], sol_x[2,:], color="r", alpha= 1)

    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    #surf = ax.plot_surface(u, v, z, cmap=plt.cm.coolwarm,
    #                   linewidth=0, antialiased=False)
    ax.set_axis_off()
    ax.view_init(50,30,0)
    #plt.savefig('2-hyperbolic_h_denoising_sig_0.3000.pdf',dpi=400)