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
from itertools import product, combinations


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

def sample_grassmannian(len,dim,kap1,kap2):
    Noise1, Data1 = sample_vMF_signal(len,dim,kap1)
    Noise2a, Data2a = sample_vMF_signal(len+1,dim,kap2)
    Noise2a = Noise2a[:,10:len*20+10]
    Data2a = Data2a[:,10:len*20+10]

    Noise2 = Noise2a - np.sum(Noise2a*Noise1,0)*Noise1
    Noise2 = Noise2/np.sqrt(np.sum(Noise2**2,0))
    Data2 = Data2a - np.sum(Data2a*Data1,0)*Data1
    Data2 = Data2/np.sqrt(np.sum(Data2**2,0))

    Noise = np.zeros((dim,2,20*len))
    Data = np.zeros((dim,2,20*len))
    Noise[:,0,:] = Noise1
    Noise[:,1,:] = Noise2
    Data[:,0,:] = Data1
    Data[:,1,:] = Data2

    DData = np.zeros((2,2,20*len))
    for l in range(20*len):
        DData[:,:,l] = Data[:,:,l].T@Data[:,:,l]
    print('orthogonal test: ', np.abs(np.sum(DData) - 2*len*20))

    return Noise, Data

def ort_grassmannian(data):
    d,k,len = np.shape(data)
    tr = np.random.randn(d,k,len)
    tr[:,0,:] = tr[:,0,:] - np.sum(tr[:,0,:]*data[:,0,:])*data[:,0,:] - np.sum(tr[:,0,:]*data[:,1,:])*data[:,1,:]
    tr[:,0,:] = tr[:,0,:]/np.sqrt(np.sum(tr[:,0,:]**2))
    tr[:,1,:] = tr[:,1,:] - np.sum(tr[:,1,:]*data[:,0,:])*data[:,0,:] - np.sum(tr[:,1,:]*data[:,1,:])*data[:,1,:] - np.sum(tr[:,1,:]*tr[:,0,:])*tr[:,0,:]
    tr[:,1,:] = tr[:,1,:]/np.sqrt(np.sum(tr[:,1,:]**2))
    tr[:,2:4,:] = data

    for l in range(len):
        Q, R = np.linalg.qr(tr[:,:,l])
        tr[:,:,l] = R
    
    return tr




#############################################
#
# operators
#
#############################################

def L_red(X,LF,N,d,k):

    L = np.zeros((d+2*k,d+2*k,N-1))

    L[d:d+k,0:d,:] = np.transpose(X[:,:,0:N-1], (1, 0, 2))
    L[d+k:d+2*k,0:d,:] = np.transpose(X[:,:,1:N], (1, 0, 2))

    L[0:d,d:d+k,:] = X[:,:,0:N-1]
    L[0:d,d+k:d+2*k,:] = X[:,:,1:N]

    L[d:d+k,d+k:d+2*k,:] = LF
    L[d+k:d+2*k,d:d+k,:] = LF

    return L

def Ltv_red(X,N,d,k):

    L = np.zeros((d+k,d+k,N))

    L[d:d+k,0:d,:] = np.transpose(X, (1, 0, 2))
    L[0:d,d:d+k,:] = X

    return L

def adjLtv_red(U,N,d,k):

    X = np.zeros((d,k,N))

    X= 2*U[0:d,d:d+k,:]

    return X

def adjL_red(U,N,d,k):

    X = np.zeros((d,k,N))

    X[:,:,0:N-1] = 2*U[0:d,d:d+k,:]

    X[:,:,1:N] = X[:,:,1:N] + 2*U[0:d,d+k:d+2*k,:]

    LF = 2*U[d:d+k,d+k:d+2*k,:]

    return [X,LF]

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
		
		[D, V] = np.linalg.eigh(L[:,:,i])
		
		D = np.diag(np.maximum(np.real(D),-1))

		L[:,:,i] = V@D@np.transpose(np.conjugate(V))

	return L

def ADMMTVprox(L, N):

	'''
	proximity operator for the real-valued model
	'''

	for i in range(N):
		
		U, S, Vt = np.linalg.svd(L[:,:,i], full_matrices=False)
		
		S = np.minimum(S,1)

		L[:,:,i] = U@np.diag(S)@Vt

	return L



#############################################
#
# solvers
#
#############################################

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


def ADMM_red(Y, Y0, lam, rho, iter):

    ''' 
    rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
    the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
    Hence, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    d, k, N = np.shape(Y)
    X = np.zeros((d,k,N), dtype='float64')
    XX = np.zeros((k,k,N), dtype='float64')
    LF = np.zeros((k,k,N-1), dtype='float64')
    #X = Y
    #for l in range(N):
    #    XX[:,:,l] = X[:,:,l].T@X[:,:,l]
    #LF = XX

    Z = np.zeros((d+2*k,d+2*k,N-1), dtype='float64')
    U = np.zeros((d+2*k,d+2*k,N-1), dtype='float64')

    data = np.zeros(iter, dtype='float64')
    datatime = np.zeros(iter, dtype='float64')
    qressphere = np.zeros(iter, dtype='float64')

    flaggg = 0
    flagggg = 0

    print('iteration \t| func-value \t| sherical-error \t| abs-error \t| orthogonal-error')
    print('--------------------------------------------------------------------------')

    starttime = time.time()
    
    for i in range(iter):

        [adjZx, adjZFL] = adjL_red(Z, N, d, k)
        [adjUx, adjUFL] = adjL_red(U, N, d, k)

        #s <- argmin_s f(s) + rho/2*||Ls - u + z||^2  ------- first ADMM-step
        X[:,:,0] = 1/2*(adjUx[:,:,0] - adjZx[:,:,0] + 1/rho*Y[:,:,0])
        X[:,:,1:N-1] = 1/4*(adjUx[:,:,1:N-1] - adjZx[:,:,1:N-1] + 1/rho*Y[:,:,1:N-1])
        X[:,:,N-1] = 1/2*(adjUx[:,:,N-1] - adjZx[:,:,N-1] + 1/rho*Y[:,:,N-1])
        LF = 1/2*(adjUFL - adjZFL)
        for j in range(N-1):
            np.fill_diagonal(LF[:,:,j], 1/2*(np.diag(adjUFL[:,:,j]) - np.diag(adjZFL[:,:,j]) + 1/rho*lam))

        #U <- argmin (.) = prox_{hpsd + I => 0}(.)  ------- second ADMM-step
        temp = L_red(X, LF, N, d, k)
        Utemp = U
        U = ADMMprox(temp.copy() + Z, N)

        #Z <- Z + Ls - U  ------- third ADMM-step // update
        Z += temp - U

        flag = 'unsphered'

        data[i] =  np.sum(1 - np.sum(X*Y,0)) + lam*np.sum(1 - LF.diagonal(offset=0, axis1=0, axis2=1))

        datatime[i] = time.time()
        qressphere[i] = 1 - np.mean(np.sqrt(np.sum(X**2,0)))
        qressphere[i] = np.mean(np.abs(1 - np.sqrt(np.sum(X**2,0))))

        if np.mod(i,100) == 0:
            if np.linalg.norm(1 - np.sqrt(np.sum(X**2,0))) < 1e-6:
                flag = 'sphered'
            
            for l in range(N):
                XX[:,:,l] = X[:,:,l].T@X[:,:,l]
                np.fill_diagonal(XX[:,:,l], np.zeros(k))
            print( i , '\t\t|', "%10.3e"% (data[i]) , '\t|', "%10.3e"% (1 - np.mean(np.sum(X**2,0))), '\t\t|', "%10.3e"% np.linalg.norm(X - Y0), '\t|', "%10.3e"% (np.sum(np.abs(XX))))

    return [X, LF, qressphere, datatime - starttime, data]


def ADMM_TV(Y, Y0, mu, rho, iter):

    ''' 
    rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
    the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
    Hence, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    d, k, N = np.shape(Y)
    X = np.zeros((d+k,d+k,N), dtype='float64')
    XX = np.zeros((k,k,N), dtype='float64')
    YL = Ltv_red(Y,N,d,k)
    Y0L = Ltv_red(Y0,N,d,k)
    #X = Y
    #for l in range(N):
    #    XX[:,:,l] = X[:,:,l].T@X[:,:,l]

    Z = np.zeros((d+k,d+k,N), dtype='float64')
    U = np.zeros((d+k,d+k,N), dtype='float64')

    data = np.zeros(iter, dtype='float64')
    datatime = np.zeros(iter, dtype='float64')
    qressphere = np.zeros(iter, dtype='float64')

    flaggg = 0
    flagggg = 0

    print('iteration \t| func-value \t| sherical-error \t| abs-error \t| orthogonal-error')
    print('------------------------------------------------------------------------------------------')

    starttime = time.time()
    
    for i in range(iter):

        X11 = np.copy(X)

        # argmin_x  -<x,y> + mu|Dx|_1 + iota(u) + rho/2||x - u + z||_2
        for l in range(0,d):
            for ll in range(d,d+k):
                X[l,ll,:] = condat_tv.tv_denoise(U[l,ll,:] - Z[l,ll,:] + YL[l,ll,:]/2/rho, mu/rho)
                X[ll,l,:] = X[l,ll,:]

        flag = (1 - np.mean(np.sum(X**2,0)))

        # proj_B(1)
        U = ADMMprox(X + Z, N+1)
        U = adjLtv_red(U,N,d,k)/2
        U = Ltv_red(U,N,d,k)

        # update
        Z = Z + X - U

        data[i] =  np.sum(1 - 1/2*np.sum(X*YL,0)) + mu/2*np.sum(np.abs(X[:,:,0:N-1] - X[:,:,1:N]))

        datatime[i] = time.time()
        qressphere[i] = 1 - np.mean(np.sqrt(np.sum(X**2,0)))
        qressphere[i] = np.mean(np.abs(1 - np.sqrt(np.sum(X**2,0))))

        if np.mod(i,100) == 0:
            if np.linalg.norm(1 - np.sqrt(np.sum(X**2,0))) < 1e-6:
                flag = 'sphered'
            
            for l in range(N):
                XX[:,:,l] = X[0:d,d:d+k,l].T@X[0:d,d:d+k,l]
                np.fill_diagonal(XX[:,:,l], np.zeros(k))
            print( i , '\t\t|', "%10.3e"% (data[i]) , '\t|', "%10.3e"% (1 - 1/2*np.mean(np.sum(X[0:d,d:d+k,:]**2,0))), '\t\t|', "%10.3e"% np.linalg.norm(X - Y0L), '\t|', "%10.3e"% (np.sum(np.abs(XX))))

    return [adjLtv_red(X,N,d,k)/2, qressphere, datatime - starttime, data]


def ADMM_TV_red(Y, Y0, mu, rho, iter):

    ''' 
    rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
    the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
    Hence, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    d, k, N = np.shape(Y)
    X = np.zeros((d,k,N), dtype='float64')
    XX = np.zeros((k,k,N), dtype='float64')
    #X = Y
    #for l in range(N):
    #    XX[:,:,l] = X[:,:,l].T@X[:,:,l]

    Z = np.zeros((d,k,N), dtype='float64')
    U = np.zeros((d,k,N), dtype='float64')

    data = np.zeros(iter, dtype='float64')
    datatime = np.zeros(iter, dtype='float64')
    qressphere = np.zeros(iter, dtype='float64')

    flaggg = 0
    flagggg = 0

    print('iteration \t| func-value \t| sherical-error \t| abs-error \t| orthogonal-error')
    print('--------------------------------------------------------------------------')

    starttime = time.time()
    
    for i in range(iter):

        X11 = np.copy(X)

        # argmin_x  -<x,y> + mu|Dx|_1 + iota(u) + rho/2||x - u + z||_2
        for l in range(d):
            for ll in range(k):
                X[l,ll,:] = condat_tv.tv_denoise(U[l,ll,:] - Z[l,ll,:] + Y[l,ll,:]/rho, mu/rho)

        flag = (1 - np.mean(np.sum(X**2,0)))

        # proj_B(1)
        U = ADMMTVprox(X + Z, N)

        # update
        Z = Z + X - U

        data[i] =  np.sum(1 - np.sum(X*Y,0)) + mu*np.sum(np.abs(X[:,:,0:N-1] - X[:,:,1:N]))

        datatime[i] = time.time()
        qressphere[i] = 1 - np.mean(np.sqrt(np.sum(X**2,0)))
        qressphere[i] = np.mean(np.abs(1 - np.sqrt(np.sum(X**2,0))))

        if np.mod(i,100) == 0:
            if np.linalg.norm(1 - np.sqrt(np.sum(X**2,0))) < 1e-6:
                flag = 'sphered'
            
            for l in range(N):
                XX[:,:,l] = X[:,:,l].T@X[:,:,l]
                np.fill_diagonal(XX[:,:,l], np.zeros(k))
            print( i , '\t\t|', "%10.3e"% (data[i]) , '\t|', "%10.3e"% (1 - np.mean(np.sum(X**2,0))), '\t\t|', "%10.3e"% np.linalg.norm(X - Y0), '\t|', "%10.3e"% (np.sum(np.abs(XX))))

    return [X, qressphere, datatime - starttime, data]

#############################################
#
# plots
#
#############################################

def ort_grass(Data):
    pData = np.zeros(np.shape(Data))
    p = np.array([1.0, 0.0, 0.0])
    len = np.shape(Data)[2]
    for i in range(len):
        pData[:,0,i] = Data[:,0,i] - np.sum(Data[:,0,i]*p) * p
        pData[:,0,i] /= np.linalg.norm(pData[:,0,i])
        pData[:,1,i] = Data[:,1,i] - np.sum(Data[:,1,i]*p) * p - np.sum(Data[:,1,i]*pData[:,0,i]) * pData[:,0,i]
        pData[:,1,i] /= np.linalg.norm(pData[:,1,i])
    return pData

def plot_grassmannian(Data, Noise, Sol):
    # Generate dummy orthonormal frame in S^3 for demonstration
    T = np.shape(Data)[2]
    signal = Data


    # Plotting
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(131, projection='3d')

    offset = 3  # spacing between time steps along x-axis

    proj = ort_grass(Data)

    for t in range(T):

        origin = np.array([offset * t, 0, 0])  # move origin along x-axis

        # Draw both vectors in the frame
        for i in range(2):
            ax1.quiver(*origin, *proj[:, i, t], color='C'+str(i),
                    length=1.0, normalize=True, alpha=0.2)

    # Adjust limits
    ax1.set_xlim([-1, offset * T])
    ax1.set_ylim([-2, 2])
    ax1.set_zlim([-2, 2])
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.zaxis.set_ticklabels([])

    ax2 = fig.add_subplot(132, projection='3d')

    proj = ort_grass(Noise)

    for t in range(T):

        origin = np.array([offset * t, 0, 0])  # move origin along x-axis

        # Draw both vectors in the frame
        for i in range(2):
            ax2.quiver(*origin, *proj[:, i, t], color='C'+str(i),
                    length=1.0, normalize=True, alpha=0.2)

    # Adjust limits
    ax2.set_xlim([-1, offset * T])
    ax2.set_ylim([-2, 2])
    ax2.set_zlim([-2, 2])
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.zaxis.set_ticklabels([])

    ax3 = fig.add_subplot(133, projection='3d')

    offset = 3  # spacing between time steps along x-axis

    proj = ort_grass(Sol)

    for t in range(T):

        origin = np.array([offset * t, 0, 0])  # move origin along x-axis

        # Draw both vectors in the frame
        for i in range(2):
            ax3.quiver(*origin, *proj[:, i, t], color='C'+str(i),
                    length=1.0, normalize=True, alpha=0.2)

    # Adjust limits
    ax3.set_xlim([-1, offset * T])
    ax3.set_ylim([-2, 2])
    ax3.set_zlim([-2, 2])
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.zaxis.set_ticklabels([])

    plt.tight_layout()
    plt.show()

def plot_grassmannian_un(Data, Noise, Sol):
    # Generate dummy orthonormal frame in S^3 for demonstration
    T = np.shape(Data)[2]

    # Plotting
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')

    offset = 10  # spacing between time steps along x-axis

    proj = ort_grass(Data)

    for t in range(T):

        origin = np.array([offset * t, 0, 0])  # move origin along x-axis

        # Draw both vectors in the frame
        for i in range(2):
            ax1.quiver(offset * t, 0, 0, Data[0, i, t], Data[1, i, t], Data[2, i, t], color='C'+str(i), alpha=0.2)

    # Adjust limits
    ax1.set_xlim([-1, offset * T])
    ax1.set_ylim([-2, 2])
    ax1.set_zlim([-2, 2])
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.zaxis.set_ticklabels([])

    ax2 = fig.add_subplot(132, projection='3d')

    proj = ort_grass(Noise)

    for t in range(T):

        origin = np.array([offset * t, 0, 0])  # move origin along x-axis

        # Draw both vectors in the frame
        for i in range(2):
            ax2.quiver(offset * t, 0, 0, Noise[0, i, t], Noise[1, i, t], Noise[2, i, t], color='C'+str(i), alpha=0.2)

    # Adjust limits
    ax2.set_xlim([-1, offset * T])
    ax2.set_ylim([-2, 2])
    ax2.set_zlim([-2, 2])
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.zaxis.set_ticklabels([])

    ax3 = fig.add_subplot(133, projection='3d')

    proj = ort_grass(Sol)

    for t in range(T):

        origin = np.array([offset * t, 0, 0])  # move origin along x-axis

        # Draw both vectors in the frame
        for i in range(2):
            ax3.quiver(offset * t, 0, 0, Sol[0, i, t], Sol[1, i, t], Sol[2, i, t], color='C'+str(i), alpha=0.2)

    # Adjust limits
    ax3.set_xlim([-1, offset * T])
    ax3.set_ylim([-2, 2])
    ax3.set_zlim([-2, 2])
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.zaxis.set_ticklabels([])

    plt.tight_layout()

def plot_grassmannian_val(Data, Noise, Sol):
    # Generate dummy orthonormal frame in S^3 for demonstration
    T = np.shape(Data)[2]

    cmap = plt.cm.get_cmap('twilight')

    # Stereographic projection from R^4 -> R^3
    def stereographic_proj(x):
        ar = np.arccos(np.sum(np.ones(4)*x))
        return [0,np.cos(ar), np.sin(ar)]
    def stereographic_val(x):
        ar = np.arctan2(np.array([1,1,-1,-1]),x)
        return ar

    # Plotting
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')

    offset = 1  # spacing between time steps along x-axis

    for t in range(T):
        if np.mod(t,3)==0:
            frame = Data[:, :, t]           # shape (4, 2)
            proj = stereographic_proj(frame[:,1])  # shape (3, 2)

            origin = np.array([offset * t, 0, 0])  # move origin along x-axis

            # Draw both vectors in the frame
            color = cmap((np.pi+stereographic_val(frame[:,0]))/(2*np.pi))[0]
            ax1.quiver(*origin, *proj, color=[color],
                        length=1.0, normalize=True)

    # Adjust limits
    ax1.set_xlim([-1, offset * T])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([0, 1])
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.zaxis.set_ticklabels([])

    ax2 = fig.add_subplot(132, projection='3d')

    offset = 1  # spacing between time steps along x-axis

    for t in range(T):
        if np.mod(t,3)==0:
            frame = Noise[:, :, t]           # shape (4, 2)
            proj = stereographic_proj(frame[:,1])  # shape (3, 2)

            origin = np.array([offset * t, 0, 0])  # move origin along x-axis

            # Draw both vectors in the frame
            color = cmap((np.pi+stereographic_val(frame[:,0]))/(2*np.pi))[0]
            ax2.quiver(*origin, *proj, color=[color],
                        length=1.0, normalize=True)

    # Adjust limits
    ax2.set_xlim([-1, offset * T])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([0, 1])
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    ax2.zaxis.set_ticklabels([])


    ax3 = fig.add_subplot(133, projection='3d')

    offset = 1  # spacing between time steps along x-axis

    for t in range(T):
        if np.mod(t,3)==0:
            frame = Sol[:, :, t]           # shape (4, 2)
            proj = stereographic_proj(frame[:,1])  # shape (3, 2)

            origin = np.array([offset * t, 0, 0])  # move origin along x-axis

            # Draw both vectors in the frame
            color = cmap((np.pi+stereographic_val(frame[:,0]))/(2*np.pi))[0]
            ax3.quiver(*origin, *proj, color=[color],
                        length=1.0, normalize=True)

    # Adjust limits
    ax3.set_xlim([-1, offset * T])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([0, 1])
    ax3.xaxis.set_ticklabels([])
    ax3.yaxis.set_ticklabels([])
    ax3.zaxis.set_ticklabels([])

    plt.tight_layout()

def angle_SO3(Data):

    for i in range(np.size(Data[0,:])):
        if Data[0,i]<0:
            Data[:,i] = -Data[:,i]
            
    alpha = np.arctan2(np.sqrt(Data[1,:]**2+Data[2,:]**2+Data[3,:]**2), Data[0,:])
    v = np.array([Data[1,:], Data[2,:], Data[3,:]]/np.sqrt(Data[1,:]**2+Data[2,:]**2+Data[3,:]**2))
    theta = np.arccos(v[2])
    phi = np.arctan2(Data[2,:], Data[1,:])

    return [alpha, phi, theta]

def plot_grassmannian_camera(Data, Noise, Sol1, Sol2):

    k = 50
    l = 50

    k = 19
    l = 10

    fig = plt.figure(figsize=(3,10), dpi=300)

    for i in range(20):
        cam2world = pt.transform_from_pq([-0.75, -2.0+i/4.5, 0, Data[0,k*i+l], Data[1,k*i+l], Data[2,k*i+l], Data[3,k*i+l]])
        
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

        cam2world = pt.transform_from_pq([0, -2.0+i/4.5, 0, Noise[0,k*i+l], Noise[1,k*i+l], Noise[2,k*i+l], Noise[3,k*i+l]])
        
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
        
        cam2world = pt.transform_from_pq([0.77, -2.0+i/4.5, 0, Sol1[0,k*i+l], Sol1[1,k*i+l], Sol1[2,k*i+l], Sol1[3,k*i+l]])

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
        
        cam2world = pt.transform_from_pq([1.55, -2.0+i/4.5, 0, Sol2[0,k*i+l], Sol2[1,k*i+l], Sol2[2,k*i+l], Sol2[3,k*i+l]])

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
    ax.set_box_aspect((0.5,0.5,1.1))
    #ax.view_init(azim=120, elev=20)
    plt.axis('off')