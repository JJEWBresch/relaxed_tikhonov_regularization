'''
Creator: Jonas Bresch, M.Sc.
Project: Bar- and QR-code denoising

Date: Nov. 23th, 2023
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import condat_tv
import cvxpy as cp


def truncate(num, n):
    integer = np.array(num * (10**n), dtype=int)/(10**n)
    return np.array(integer, dtype=float)

def MIoU_cut(xT, xS):

    n = np.shape(xT)

    s = np.max(np.abs(xS))

    xT = np.array(truncate((xT+s), 2)*100, dtype=int)
    xS = np.array(truncate((xS+s), 2)*100, dtype=int)

    count_xT = np.array([np.count_nonzero(xT == k) for k in range(np.max(xS)+1)], dtype=int)
    count_xS = np.array([np.count_nonzero(xS == k) for k in range(np.max(xS)+1)], dtype=int)

    num_clas = np.size(count_xS)
    #num_clas = np.max(xT)+1

    vec_xT = np.reshape(np.array(xT, dtype=int), n)
    vec_xS = np.reshape(np.array(xS, dtype=int), n)


    cat = num_clas*vec_xT + vec_xS


    count_cat_vec = np.array([np.count_nonzero(cat == k) for k in range(num_clas**2)])


    count_cat_mat = count_cat_vec.reshape(num_clas, num_clas)

    I = np.diag(count_cat_mat)

    U = count_xT + count_xS - I

    indivIoU = np.nan*np.ones(np.size(I))

    for i in range(np.size(U)):
        if U[i] != 0:
            indivIoU[i] = I[i]/U[i]

    return indivIoU

def MIoU(xT, xS):

    n = np.shape(xT)

    xT = np.array(np.round((xT+1)/2), dtype=int)
    xS = np.array(np.round((xS+1)/2), dtype=int)

    count_xT = np.array([np.count_nonzero(xT == k) for k in range(np.max(xS)+1)], dtype=int)
    count_xS = np.array([np.count_nonzero(xS == k) for k in range(np.max(xS)+1)], dtype=int)

    num_clas = np.size(count_xS)
    #num_clas = np.max(xT)+1

    vec_xT = np.reshape(np.array(xT, dtype=int), n)
    vec_xS = np.reshape(np.array(xS, dtype=int), n)


    cat = num_clas*vec_xT + vec_xS


    count_cat_vec = np.array([np.count_nonzero(cat == k) for k in range(num_clas**2)])


    count_cat_mat = count_cat_vec.reshape(num_clas, num_clas)

    I = np.diag(count_cat_mat)

    U = count_xT + count_xS - I

    indivIoU = I/U

    return indivIoU

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

######## SIGNAL-GENERATOR ########

def barcode_generator(lam,n,scale):

    ''' yields us the tangential normal distribution on S^0 = {-1, 1} 
        where the tangential hyperplanes T_{-1}S^0 = T_{1}S^0 = < 1 > = R. 
    '''

    data_0 = np.zeros(n)
    data_0[0] = 1
    for i in range(1,n):
        diff = np.random.randn(1)
        if np.sign(diff)<0:
            data_0[i] = -1
        else:
            data_0[i] = 1
    d_0 = data_0

    data_0 = np.random.randn(n)
    d_0 = np.sign(data_0)
    d_0[0] = d_0[n-1] = 1

    ''' with standard deviation : sigma = sqrt(var) = sqrt(lam/100) = 1/10*sqrt(lam)
    '''
    d = d_0 + np.random.randn(n)*np.sqrt(lam)

    ''' 10-times finer signal with noise
    '''

    x = np.arange(n)
    #scale = 10
    dx = np.arange(scale+1)

    def piecewise_linear(x, x0, y0, m):
        eta = x/m
        return (1-eta)*x0 + eta*y0

    dd_0 = []
    for i in range(n-1):
        dd_0 = np.append(dd_0, piecewise_linear(dx, d_0[i], d_0[i+1], scale))

    dd_0 = []
    for i in range(n-1):
        dd_0 = np.append(dd_0, np.ones(scale)*d_0[i])

    dd = dd_0 + np.random.randn(np.size(dd_0))*np.sqrt(lam)

    print('standard deviation : sigma =', np.sqrt(lam))

    plt.figure(1,figsize=(15,3))
    plt.plot(dd_0, 'b')
    plt.plot(dd, 'k', linewidth=0.5)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    
    return [dd, dd_0]

def gen_sig_counterex(n, sig):

    dd_0 = np.zeros(n)

    for i in range(n):
        if np.mod(i,2) == 0:
            dd_0[i] = 1
        else:
            dd_0[i] = -1
    
    dd = dd_0 + np.sqrt(sig)*np.random.randn(n)

    ddconst = dd_0*(1+sig)

    plt.figure(1,figsize=(15,3))
    plt.plot(dd_0, 'b')
    plt.plot(dd, 'k', linewidth=0.5)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)

    return [dd, dd_0, ddconst]

def qr_code_generator(n,sig,scale):

    lam = (np.sqrt(2)*sig)**2
    kap = 0.6
    #n = 20
    #scale = 10
    N = n
    ''' yields us the tangential normal distribution on S^0 = {-1, 1} 
        where the tangential hyperplanes T_{-1}S^0 = T_{1}S^0 = < 1 > = R. 
    '''

    x = y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')

    data_0 = np.zeros((n,n), dtype=complex)
    data_0 = np.random.randn(n,n)
    data_0 = np.random.rand(n,n)

    d_0 = np.sign(data_0-0.25)

    data_0 = np.random.uniform(-1,1,(n,n))

    d_0 = np.sign(data_0)

    dd_0 = np.zeros((n*scale, n*scale))
    for i in range(n):
        for j in range(n):
            dd_0[scale*i:scale*i+scale, scale*j:scale*j+scale] = np.ones((scale,scale))*d_0[i,j]

    ''' with standard deviation : sigma = sqrt(var) = sqrt(lam/100) = 1/10*sqrt(lam)
    '''

    d = d_0 + np.random.randn(n,n)*np.sqrt(lam)

    dd = dd_0 + np.random.randn(scale*n,scale*n)*np.sqrt(lam)

    print('standard deviation : sigma =', np.sqrt(lam))

    '''cd_0 = (d_0-np.min(d_0))/(np.max(d_0)-np.min(d_0))

    plt.figure(0)
    plt.imshow(d_0, alpha = cd_0, cmap='Greys')
    plt.savefig("0D_2D_ground_truth.png",dpi=400)

    cd = (d-np.min(d))/(np.max(d)-np.min(d))

    plt.figure(1)
    plt.imshow(d, alpha = cd, cmap='Greys')
    plt.savefig("0D_2D_noise.png",dpi=400)

    cdd = (dd-np.min(dd))/(np.max(dd)-np.min(dd))

    plt.figure(1)
    plt.imshow(dd, alpha = cdd, cmap='Greys')
    plt.savefig("0D_2D_noise_scale.png",dpi=400)'''

    return [dd, dd_0]


######## HELP-FUNCTIONS #########

def L(x,r,N):

    U = np.zeros((N-1,3,3))

    U[:,1,0] = x[0:N-1]
    U[:,0,1] = np.conjugate(x[0:N-1])

    U[:,2,0] = x[1:N]
    U[:,0,2] = np.conjugate(x[1:N])

    U[:,1,2] = r
    U[:,2,1] = np.conjugate(r)

    return U

def L_image(x,r1,r2,N):

    #N = np.size(x[:,0])

    U1 = np.zeros((N-1, N, 3,3))

    U1[:,:,1,0] = x[0:N-1,:]
    U1[:,:,0,1] = x[0:N-1,:]
    U1[:,:,2,0] = x[1:N,:]
    U1[:,:,0,2] = x[1:N,:]
    U1[:,:,1,2] = r1
    U1[:,:,2,1] = r1

    U2 = np.zeros((N, N-1, 3,3))

    U2[:,:,1,0] = x[:,0:N-1]
    U2[:,:,0,1] = x[:,0:N-1]
    U2[:,:,2,0] = x[:,1:N]
    U2[:,:,0,2] = x[:,1:N]
    U2[:,:,1,2] = r2
    U2[:,:,2,1] = r2 

    return [U1, U2]

def adjL(U,N):
	x = np.zeros(N)

	x[0:N-1] = U[:,1,0] + np.matrix.conj(U[:,0,1])
	x[1:N] = x[1:N] + np.squeeze(U[:,2,0] + np.matrix.conj(U[:,0,2]))
	r = np.squeeze(U[:,1,2] + np.matrix.conj(U[:,2,1]))

	return [x,r]

def adjL_image(U1, U2, N):

    x = np.zeros((N, N))

    x[0:N-1,:] = U1[:,:,1,0] + U1[:,:,0,1]
    x[1:N,:] = x[1:N,:] + np.squeeze(U1[:,:,2,0] + U1[:,:,0,2])
    r1 = np.squeeze(U1[:,:,1,2] + U1[:,:,2,1])

    x[:,0:N-1] = x[:,0:N-1] + U2[:,:,1,0] + U2[:,:,0,1]
    x[:,1:N] = x[:,1:N] + np.squeeze(U2[:,:,2,0] + U2[:,:,0,2])
    r2 = np.squeeze(U2[:,:,1,2] + U2[:,:,2,1])

    return [x, r1, r2]

def L_R(x,r,N):

    U = np.zeros((N-1,4,4))

    U[:,2,0] = np.real(x[0:N-1])
    U[:,2,1] = np.imag(x[0:N-1])
    U[:,0,2] = np.real(x[0:N-1])
    U[:,1,2] = np.imag(x[0:N-1])

    U[:,3,0] = np.real(x[1:N])
    U[:,3,1] = np.imag(x[1:N])
    U[:,0,3] = np.real(x[1:N])
    U[:,1,3] = np.imag(x[1:N])

    U[:,2,3] = r
    U[:,3,2] = r

    return U

def adjL_R(U,N):

    x = np.zeros(N)

    x[0:N-1] = U[:,2,0] + np.matrix.conj(U[:,0,2])
    x[0:N-1] = x[0:N-1] + 1j*(U[:,2,1] + np.matrix.conj(U[:,1,2]))
    x[1:N] = x[1:N] + np.squeeze(U[:,3,0] + np.matrix.conj(U[:,0,3]))
    x[1:N] = x[1:N] + 1j*(np.squeeze(U[:,3,1] + np.matrix.conj(U[:,1,3])))

    r = np.squeeze(U[:,2,3] + np.matrix.conj(U[:,3,2]))

    return [x,r]

def L_1R(x,r,N):

    U = np.zeros((N-1,3,3))

    U[:,1,0] = np.real(x[0:N-1])
    U[:,0,1] = np.real(x[0:N-1])

    U[:,2,0] = np.real(x[1:N])
    U[:,0,2] = np.real(x[1:N])

    U[:,1,2] = r
    U[:,2,1] = r

    return U

def adjL_1R(U,N):

    x = np.zeros(N)

    x[0:N-1] = U[:,1,0] + U[:,0,1]
    x[1:N] = x[1:N] + np.squeeze(U[:,2,0] + U[:,0,2])
    r = np.squeeze(U[:,1,2] + U[:,2,1])

    return [x,r]

def ADMMprox(U, Z, N):

	for i in range(0,N-1):
		
		[D, V] = np.linalg.eig(U[i,:,:] + Z[i,:,:])

		D = np.diag(np.maximum(np.real(D),-1))

		U[i,:,:] = V@D@np.transpose(np.conjugate(V))

	return U

def ADMMprox_image(U1, U2, rho, N):

    for i in range(0,N-1):
        for j in range(0,N):
            [D, V] = np.linalg.eig(U1[i,j,:,:])
            D = np.diag(np.maximum(np.real(D),-1))
            U1[i,j,:,:] = V@D@np.transpose(np.conjugate(V))
    
    for k in range(0,N):
        for l in range(0,N-1):
            [D, V] = np.linalg.eig(U2[k,l,:,:])
            D = np.diag(np.maximum(np.real(D),-1))
            U2[k,l,:,:] = V@D@np.transpose(np.conjugate(V))

    return [U1, U2]

######## DENOISING-ALGORITHMS #########

def ADMM_1R(y, y0, lam, rho, iter):

    ''' rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
        the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
        Hnece, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    N = np.size(y)
    x = np.zeros(N)
    s = np.zeros(N)
    r = np.zeros(N-1)
    #x = y
    #r = y[0:N-1]*np.conjugate(y[1:N])

    Z = np.zeros((N-1,3,3))
    U = np.zeros((N-1,3,3))

    data = np.zeros(iter+1)
    datatime = np.zeros(iter+1)

    flag = 1
    i = 0

    print('iter. \t| funv-value \t\t| RMSE \t\t| spherical-error')
    print('-----------------------------------------------------------------')
    
    #while flag > 10**(-9):
    while i < iter:

        [adjZx, adjZr] = adjL_1R(Z, N)
        [adjUx, adjUr] = adjL_1R(U, N)

        #s <- argmin_s f(s) + rho/2*||Ls - u + z||^2
        x[0] = 1/(2)*(adjUx[0] - adjZx[0] + 1/rho*y[0])
        x[1:N-1] = 1/(4)*(adjUx[1:N-1] - adjZx[1:N-1] + 1/rho*y[1:N-1])
        x[N-1] = 1/(2)*(adjUx[N-1] - adjZx[N-1] + 1/rho*y[N-1])
        r = 1/2*(adjUr - adjZr + 1/rho*lam)

        X = L_1R(x, r, N)

        #U <- argmin (.) = prox_{hpsd + I => 0}(.)
        U = ADMMprox(L_1R(x, r, N), Z, N)

        #Z <- Z + Ls - U
        Z = Z + L_1R(x, r, N) - U

        data[i] =  np.linalg.norm(x - y)**2+lam*np.linalg.norm(x[0:N-1] - x[1:N])**2

        datatime[i] = time.time()

        flag = np.linalg.norm(np.abs(x) - 1)

        if np.mod(i,50) == 0:

            print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))
        
        i = i + 1

    data[i] =  np.linalg.norm(x - y)**2+lam*np.linalg.norm(x[0:N-1] - x[1:N])**2
    
    print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))


    diff_x = x - y0
    nx = x/np.abs(x)
    s = MIoU(y0, x)
    s = np.nanmean(s)

    datei = open('data_0D_1Dgrid_admm1R.txt','a')
    for w in range(np.size(data)):
        datei.write(str(data[w]))
        datei.write('\n')
    datei.write(str(np.sqrt(np.sum((x - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(x - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')
    datei.close()

    return [x, data]

def ADMM_1R_TV(y, y0, lam, mu, rho, iter):

    ''' rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
        the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
        Hnece, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    N = np.size(y)
    x = np.zeros(N)
    s = np.zeros(N)
    r = np.zeros(N-1)
    q_x = np.zeros(N)
    q_r = np.zeros(N-1)
    x_z = np.zeros(N)
    r_z = np.zeros(N-1)
    #x = y
    #r = y[0:N-1]*np.conjugate(y[1:N])

    Z = np.zeros((N-1,3,3))
    U = np.zeros((N-1,3,3))

    data = np.zeros(iter)
    datatime = np.zeros(iter)

    flag = 1
    i = 0

    print('iter. \t| funv-value \t\t| RMSE \t\t| spherical-error')
    print('-----------------------------------------------------------------')
    
    while flag > 10**(-9):

        #s <- argmin_s f_1(s) + f_2(s) + rho/2*||Ls - u + z||^2
        # where f_1 = <.,c> and f_2 = ||D(G(s))||_1 and Gs = s_x = [s_x, s_r] the projection onto the signal components
        x_z1 = np.random.rand(N)
        #for k in range(100):
        while np.linalg.norm(x_z - x_z1) > 10**(-9):

            x_z1 = x_z

            x = condat_tv.tv_denoise(x_z, mu)
            r = r_z

            adj_x, adj_r = adjL_1R(L(x, r, N) - U + Z, N)
            
            q_x = 2*x - x_z - mu*rho*adj_x + mu*y
            q_r = 2*r - r_z - mu*rho*adj_r + mu*lam

            x_z = x_z + q_x - x
            r_z = r_z + q_r - r

        #U <- argmin (.) = prox_{hpsd + I => 0}(.)
        U = ADMMprox(L_1R(x, r, N), Z, N)

        #Z <- Z + Ls - U
        Z = Z + L_1R(x, r, N) - U

        data[i] =  np.linalg.norm(x - y)**2 + lam*np.linalg.norm(x[0:N-1] - x[1:N])**2 + mu*np.sum(np.abs(x[0:N-1] - x[1:N]))

        datatime[i] = time.time()

        flag = np.linalg.norm(np.abs(x) - 1)

        if np.mod(i,50) == 0:

            print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))
        
        i = i + 1
    
    data[i] =  np.linalg.norm(x - y)**2 + lam*np.linalg.norm(x[0:N-1] - x[1:N])**2 + mu*np.sum(np.abs(x[0:N-1] - x[1:N]))
    
    print( i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ', "%10.2e"% (np.linalg.norm(1 - np.abs(x))))


    diff_x = x - y0
    nx = x/np.abs(x)
    s = MIoU(y0, x)
    s = np.nanmean(s)

    datei = open('data_0D_1Dgrid_admm1R_tv.txt','a')
    datei.write(str(np.sqrt(np.sum((x - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(x - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')
    datei.close()

    return [x, data]

def ADMM_1R_TV_strong(y, y0, lam, mu, eta, bet, rho, iter):

    ''' rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
        the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
        Hnece, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    N = np.size(y)
    x = np.zeros(N)
    #x = y
    s = np.zeros(N)
    r = np.zeros(N-1)
    q_x = np.zeros(N)
    q_r = np.zeros(N-1)
    x_z = np.zeros(N)
    r_z = np.zeros(N-1)
    #r = y[0:N-1]*np.conjugate(y[1:N])

    #Z = np.zeros((N-1,3,3))
    Z = -1*np.ones((N-1,3,3))
    U = np.zeros((N-1,3,3))

    data = np.zeros(iter)
    datatime = np.zeros(iter)

    flag = 1
    i = 0

    Z0 = Z

    print('iter. \t| funv-value \t\t| RMSE \t\t| spherical-error')
    print('-----------------------------------------------------------------')
    
    while flag > 10**(-9) and i < 30000:
    #for i in range(iter):
        #s <- argmin_s f_1(s) + f_2(s) + rho/2*||Ls - u + z||^2
        # where f_1 = <.,c> and f_2 = ||D(G(s))||_1 and Gs = s_x = [s_x, s_r] the projection onto the signal components
        x1 = np.random.rand(N)
        #for k in range(100):
        while np.linalg.norm(x - x1) > 10**(-9):
            x1 = x
            
            #x_z = condat_tv.tv_denoise(1/(mu*eta + 1)*x, mu/(mu*eta + 1))
            x_z = condat_tv.tv_denoise(1/(eta*bet + 1)*x, mu*bet/(eta*bet + 1))
            r_z = 1/(eta*bet + 1)*r
            #r_z = r

            [adjZx, adjZr] = adjL_1R(Z, N)
            [adjUx, adjUr] = adjL_1R(U, N)

            q_x[0] = bet/(2*rho*bet + 1)*(1/bet*(2*x_z[0] - x[0] + bet*eta*x_z[0]) + y[0] + rho*adjUx[0] - rho*adjZx[0])
            #q_x[0] = 1/(1 + 1/bet + 2*rho)*(1/bet*(2*x_z[0] - x[0] + bet*eta*x_z[0]) + y[0] + rho*adjUx[0] - rho*adjZx[0])
            q_x[1:N-1] = bet/(4*rho*bet + 1)*(1/bet*(2*x_z[1:N-1] - x[1:N-1] + bet*eta*x_z[1:N-1]) + y[1:N-1] + rho*adjUx[1:N-1] - rho*adjZx[1:N-1])
            #q_x[1:N-1] =  1/(1 + 1/bet + 4*rho)*(1/bet*(2*x_z[1:N-1] - x[1:N-1] + bet*eta*x_z[1:N-1]) + y[1:N-1] + rho*adjUx[1:N-1] - rho*adjZx[1:N-1])
            q_x[N-1] = bet/(2*rho*bet + 1)*(1/bet*(2*x_z[N-1] - x[N-1] + bet*eta*x_z[N-1]) + y[N-1] + rho*adjUx[N-1] - rho*adjZx[N-1])
            #q_x[N-1] = 1/(1 + 1/bet + 2*rho)*(1/bet*(2*x_z[N-1] - x[N-1] + bet*eta*x_z[N-1]) + y[N-1] + rho*adjUx[N-1] - rho*adjZx[N-1])
            q_r = bet/(2*rho*bet + 1)*(1/bet*(2*r_z - r + bet*eta*r_z) + lam + rho*adjUr - rho*adjZr)
            #q_r = 1/2/rho*(rho*adjUr - rho*adjZr)

            x = x + q_x - x_z
            r = r + q_r - r_z
        
        x = x_z
        r = r_z

        #U <- argmin (.) = prox_{hpsd + I => 0}(.)
        U = ADMMprox(L_1R(x, r, N), Z, N)

        #Z <- Z + Ls - U
        Z = Z + L_1R(x, r, N) - U

        data[i] =  np.linalg.norm(x - y)**2 + lam*np.linalg.norm(x[0:N-1] - x[1:N])**2 + mu*np.sum(np.abs(x[0:N-1] - x[1:N]))

        datatime[i] = time.time()

        flag = np.linalg.norm(np.abs(x) - 1)

        if np.mod(i,50) == 0:

            print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))
        
        i = i + 1

    data[i] =  np.linalg.norm(x - y)**2 + lam*np.linalg.norm(x[0:N-1] - x[1:N])**2 + mu*np.sum(np.abs(x[0:N-1] - x[1:N]))
    
    print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))


    diff_x = x - y0
    nx = x/np.abs(x)
    s = MIoU(y0, x)
    s = np.nanmean(s)

    datei = open('data_0D_1Dgrid_strong_tv.txt','a')
    datei.write(str(np.sqrt(np.sum((x - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(x - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')
    datei.close()

    return [x, data]

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def ADMM_1R_TV_approx(y, y0, lam, mu, eta, rho, iter, approx):

    ''' rewriting this into the max -f(x) <> min f(x) where f is confex and x in convex cone K, yields 
        the same algorithm, since the (augmented) Lagragian becomes -f(x) - g(x) added by -||...||^2 and -<.,...> 
        Hnece, just the update changed: x[0] = -1/2*(-adjUx[0] + adjZx[0] - 1/rho*y[0])
                                        x[1:N-1] = -1/4*(-adjUx[1:N-1] + adjZx[1:N-1] - 1/rho*y[1:N-1])
                                        x[N-1] = -1/2*(-adjUx[N-1] + adjZx[N-1] - 1/rho*y[N-1])
                                        r = -1/2*(-adjUr + adjZr - 1/rho*lam)
    '''

    N = np.size(y)
    x = np.zeros(N)
    #x = 1000000*np.ones(N)
    s = np.zeros(N)
    r = np.zeros(N-1)
    #r = x[0:N-1]*x[1:N]

    alp = -1/4*(4/3 - 4*np.sqrt(3))


    Z = np.zeros((N-1,3,3))
    #Z = -1*np.ones((N-1,3,3))
    U = np.zeros((N-1,3,3))

    data = np.zeros(iter)
    datatime = np.zeros(iter)

    flag = 1
    i = 0

    Z0 = Z

    print('iter. \t| funv-value \t\t| RMSE \t\t| spherical-error')
    print('-----------------------------------------------------------------')
    
    #while flag > 10**(-9):
    for i in range(iter):
        #s <- argmin_s f_1(s) + f_2(s) + rho/2*||Ls - u + z||^2
        # where f_1 = <.,c> and f_2 = ||D(G(s))||_1 and Gs = s_x = [s_x, s_r] the projection onto the signal components
        r1 = np.random.rand(N-1)
        #for k in range(100):
        while np.linalg.norm(r - r1) > 10**(-9):

            r1 = r

            if approx == 0:
                r = soft_thresh(r + 4*rho*eta/mu*((alp-r) - U[:,1,2] + Z[:,1,2]), eta)
            else:
                r = soft_thresh(r + 4*rho*eta/mu/2*((2-2*r) - U[:,1,2] + Z[:,1,2]), eta)

        if approx == 0:
            r = alp - r
        else:
            r = 2 - 2*r


        [adjZx, adjZr] = adjL_1R(Z, N)
        [adjUx, adjUr] = adjL_1R(U, N)

        x[0] = 1/rho/2*(y[0] + rho*adjUx[0] - rho*adjZx[0])
            #q_x[0] = 1/(1 + 1/bet + 2*rho)*(1/bet*(2*x_z[0] - x[0] + bet*eta*x_z[0]) + y[0] + rho*adjUx[0] - rho*adjZx[0])
        x[1:N-1] = 1/rho/4*(y[1:N-1] + rho*adjUx[1:N-1] - rho*adjZx[1:N-1])
            #q_x[1:N-1] =  1/(1 + 1/bet + 4*rho)*(1/bet*(2*x_z[1:N-1] - x[1:N-1] + bet*eta*x_z[1:N-1]) + y[1:N-1] + rho*adjUx[1:N-1] - rho*adjZx[1:N-1])
        x[N-1] = 1/rho/2*(y[N-1] + rho*adjUx[N-1] - rho*adjZx[N-1])
            #q_x[N-1] = 1/(1 + 1/bet + 2*rho)*(1/bet*(2*x_z[N-1] - x[N-1] + bet*eta*x_z[N-1]) + y[N-1] + rho*adjUx[N-1] - rho*adjZx[N-1])

        #U <- argmin (.) = prox_{hpsd + I => 0}(.)
        U = ADMMprox(L_1R(x, r, N), Z, N)

        #Z <- Z + Ls - U
        Z = Z + L_1R(x, r, N) - U

        data[i] =  np.linalg.norm(x - y)**2 + lam*np.linalg.norm(x[0:N-1] - x[1:N])**2 + mu*np.sum(np.abs(x[0:N-1] - x[1:N]))

        datatime[i] = time.time()

        flag = np.linalg.norm(np.abs(x) - 1)

        if np.mod(i,50) == 0:

            print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))
        
        #i = i + 1

    data[i] =  np.linalg.norm(x - y)**2 + lam*np.linalg.norm(x[0:N-1] - x[1:N])**2 + mu*np.sum(np.abs(x[0:N-1] - x[1:N]))
    
    print(i, ' \t| ', data[i] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(1 - np.abs(x))))

    diff_x = x - y0
    nx = x/np.abs(x)
    s = MIoU(y0, x)
    s = np.nanmean(s)

    datei = open('data_0D_1Dgrid_admm1R_tv_approx.txt','a')
    datei.write(str(np.sqrt(np.sum((x - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(x - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')
    datei.close()

    return [x, data]

######## (fast)-GME-TV #########

def h_frequence(K):

    L = 2*K-1

    h = np.zeros(L, dtype=float)
    
    for i in range(-K+1, K):
        if i == 0:
            h[K-1 + i] = 1 - 1/K
        else:
            h[K-1 + i] = (np.abs(i)/K - 1)/K
    
    return h

def g_frequence(h,K):

    L = 2*K-1

    g = np.zeros(L-1, dtype=float)
    
    for n in range(L-1):
        for k in range(n+1):
            g[n] = g[n] + h[k]
    
    return g

def G_mat(g, K, N):

    L = 2*K - 1
    G = np.zeros((N-L+1, N-1), dtype=float)

    for i in range(N-L+1):
        G[i,i:i+L-1] = g
    
    return G

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


def ista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    L = np.linalg.norm(A) ** 2  # Lipschitz constant
    time0 = time.time()
    for _ in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        this_pobj = 0.5 * np.linalg.norm(A.dot(x) - b) ** 2 + l * np.linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x

def GME_TV(y, y0, lam, iter):

    N = np.size(y)

    K = 10
    L = 2*K - 1

    x = np.zeros(N)

    h = h_frequence(K)
    g = g_frequence(h, K)

    G = G_mat(g, K, N)

    C = 1/np.sqrt(lam)*G

    data = np.zeros(iter)

    D = np.zeros((N-1,N))
    for i in range(N-1):
        D[i,i] = 1
        D[i,i+1] = -1
    
    B = C@D

    A = B.T@C

    E = B.T@B

    x1 = np.random.randn(N)
    j = 0

    print('iter. \t| funv-value \t\t| RMSE \t\t| error')
    print('-----------------------------------------------------------------')

    #for j in range(iter):
    while np.linalg.norm(x1 - x) > 10**(-5):

        x1 = x

        c = B@x

        v = ista(C, c, lam, 1000)

        z = E@x - A@v

        x = condat_tv.tv_denoise(y + lam*z, lam)

        data[j] =  np.linalg.norm(x - y)**2 + lam*np.sum(np.abs(x[0:N-1] - x[1:N]))

        if np.mod(j,50) == 0:

            print(j, ' \t| ', data[j] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(x1 - x)))
        
        j = j + 1

    data[j] =  np.linalg.norm(x - y)**2 + lam*np.sum(np.abs(x[0:N-1] - x[1:N]))

    print(j, ' \t| ', data[j] , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(x1 - x)))

    ss = MIoU(y0, np.sign(x))
    ss = np.nanmean(ss)
    s = MIoU_cut(y0, x)
    s = np.nanmean(s)

    datei = open('data_0D_1Dgrid_gme_tv.txt','a')
    datei.write(str(np.sqrt(np.sum((x - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sqrt(np.sum((np.sign(x) - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(x - y0))))
    datei.write('\n')
    datei.write(str(np.sum(np.abs(np.sign(x) - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')
    datei.write(str(ss))
    datei.write('\n')
    datei.close()

    return x

######## (fast)-ADMM-TV #########

def proj_B1(x):

        d = np.size(x)
        
        for i in range(d):
            if np.abs(x[i]) > 1:
                x[i] = x[i]/np.abs(x[i])
            
        return x

def ADMM_TV_BOX(y, y0, mu, rho, iter):

        d = np.size(y)

        x = np.zeros(d)
        u = np.zeros(d)
        z = np.zeros(d)

        print('iter. \t| funv-value \t\t| RMSE \t\t| error')
        print('-----------------------------------------------------------------')

        x1 = np.random.randn(d)

        j = 0

        while np.linalg.norm(x1 - x) > 1e-5:
            x1 = x

            # argmin_x  -<x,y> + mu|Dx|_1 + iota(u) + rho/2||x - u + z||_2
            x = condat_tv.tv_denoise(u - z + y/rho, mu/rho)

            # proj_B(1)
            u = proj_B1(x + z/rho)

            # update
            z = z + x - u
        
            print(j, ' \t| ',np.linalg.norm(x - y)**2 + mu*np.sum(np.abs(x[0:d-1] - x[1:d])) , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ' , "%10.2e"% (np.linalg.norm(x1 - x)))

            j = j+1
        
        s = MIoU(y0, x)
        s = np.nanmean(s)

        datei = open('data_0D_1Dgrid_box_tv.txt','a')
        datei.write(str(np.sqrt(np.sum((x - y0)**2))))
        datei.write('\n')
        datei.write(str(np.sum(np.abs(x - y0))))
        datei.write('\n')
        datei.write(str(s))
        datei.write('\n')
        datei.close()

        return [x, z]

def ADRA(y, gam, mu, iter):

    s = y
    x = y
    d, d = np.shape(y)
    r = np.zeros(np.shape(y))

    n = 0
    xx = np.random.rand(d,d)

    #for n in range(iter):
    #while np.linalg.norm(x - xx) > 10-5:
    while np.linalg.norm(x - xx) > 10-7:
        xx = x
        r1 = r
        r = s - x + condat_tv.tv_denoise_matrix(1/(1+gam)*(2*x - s) + gam/(1+gam)*y, mu*gam/(1+gam))
        s = r + n/(n+3)*(r - r1)
        x = condat_tv.tv_denoise_matrix(s.T , mu*gam)
        x = x.T

        n = n+1

    return x

def proj_B1_image(x):

        d, d = np.shape(x)
        
        for i in range(d):
            for j in range(d):
                #if np.linalg.norm(x[i]) > 1:
                #    x[i] = x[i]/np.linalg.norm(x[i])
                if np.abs(x[i,j]) > 1:
                    x[i,j] = x[i,j]/np.abs(x[i,j])
            
        return x

def ADMM_TV_BOX_image(y, y0, mu, rho, iter):

        d, d = np.shape(y)

        x = np.zeros((d,d))
        u = np.zeros((d,d))
        z = np.zeros((d,d))

        print('iter. \t| funv-value \t\t| RMSE \t\t| MAE \t\t| spherical-error \t| MioU')
        print('-----------------------------------------------------------------------------------------')

        x1 = np.random.randn(d,d)

        j = 0

        delta = np.linalg.norm(1 - np.abs(x))
        d1 = np.linalg.norm(1 - np.abs(x))
        d2 = np.linalg.norm(1 - np.abs(x))/np.size(x[0,:])**2

        #while np.linalg.norm(x1 - x) > 1e-3:
        while d2 > 1e-4:
            x1 = x

            # argmin_x  -<x,y> + mu|Dx|_1 + iota(u) + rho/2||x - u + z||_2
            #x = condat_tv.tv_denoise_matrix(u - z + y/rho, mu/rho)
            x = ADRA(u - z + y/rho, 1, mu/rho, 100)

            # proj_B(1)
            u = proj_B1_image(x + z)

            # update
            z = z + x - u

            s = MIoU(y0, x)
            s = np.nanmean(s)

            delta = d1 - np.linalg.norm(1 - np.abs(x))
            d1 = np.linalg.norm(1 - np.abs(x))
            d2 = np.linalg.norm(1 - np.abs(x))/np.size(x[0,:])**2
        
            if np.mod(j,50) == 0:
                print(j, ' \t| ',np.linalg.norm(x - y)**2 + mu*np.sum(np.abs(x[0:d-1,:] - x[1:d,:])) + mu*np.sum(np.abs(x[:,0:d-1] - x[:,1:d])) , ' \t| ', "%10.2e"% (np.linalg.norm(x - y0)**2), ' \t| ',  "%10.2e"% (np.sum(np.abs(x - y0))), ' \t| ' , "%10.2e"% (d2), ' \t|', "%10.2e"% s)

            j = j+1
        
        s = MIoU_cut(y0, np.sign(x))
        s = np.nanmean(s)
        
        print(j, ' \t| ',np.linalg.norm(np.sign(x) - y)**2 + mu*np.sum(np.abs(np.sign(x)[0:d-1,:] - np.sign(x)[1:d,:])) + mu*np.sum(np.abs(np.sign(x)[:,0:d-1] - np.sign(x)[:,1:d])) , ' \t| ', "%10.2e"% (np.sqrt(np.sum(np.sign(x) - y0)**2)), ' \t| ',  "%10.2e"% (np.sum(np.abs(np.sign(x) - y0))), ' \t| ' , "%10.2e"% (d2), ' \t|', "%10.2e"% s)

        datei = open('data_0D_2Dgrid_box_tv.txt','a')
        datei.write(str(np.sqrt(np.sum((np.sign(x) - y0)**2))))
        datei.write('\n')
        datei.write(str(np.sum(np.abs(np.sign(x) - y0))))
        datei.write('\n')
        datei.write(str(s))
        datei.write('\n')

        return [x, z]

######### ANISOTROPIC-TV #########

def DIFF(d):

    D = np.zeros((d-1,d))
    for i in range(d-1):
        D[i,i] = -1
        D[i,i+1] = 1
    #D[d-1,d-1] = 1
    
    return D

def AnisotropicTV(y, y0, lam):

    d = np.size(y)

    D = DIFF(d)

    et = np.ones(d-1)
    c = np.abs(1-y) - np.abs(y)

    # Construct the problem.
    x1 = cp.Variable(d-1)
    x2 = cp.Variable(d-1)
    x = cp.Variable(d)

    objective = cp.Minimize(cp.sum(x1) + cp.sum(x2) + lam * cp.sum(c.T @ x))
    #objective = cp.Minimize(cp.sum(x1) + cp.sum(x2))
    constraints = [ 0 <= x1, 0 <= x2, D @ x == x1 - x2 , x <= 1, 0 <= x]

    prob = cp.Problem(objective, constraints)

    #result = prob.solve(verbose=True)
    result = prob.solve()

    xsol = x.value
    x1sol = x1.value
    x2sol = x2.value

    # projection onto S_0 
    for i in range(d):
        if xsol[i]> 0.5:
            xsol[i] = 1
        else:
            xsol[i] = 0

    s = MIoU(2*y0-1, 2*xsol-1)
    s = np.nanmean(s)

    if np.linalg.norm(np.abs(2*xsol-1) - np.ones(np.size(xsol))) > 10**(-5):
        print('not tight')

    datei = open('data_0D_1Dgrid_aniso_tv.txt','a')
    datei.write(str(np.sqrt(4*np.sum((xsol - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(2*np.abs(xsol - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')

    return [result, xsol, x1sol, x2sol]

def trans_signal(y):

    d = np.size(y)

    y = 1/2*(1+y)

    for i in range(d):
        if y[i] < 0:
            y[i] = np.abs(y[i])
        if y[i] > 1:
            y[i] = 1 - (y[i] - 1)
        
    return y

def signum(y):
    
    d = np.size(y)
    for i in range(d):
        if y[i]> 0.0:
            y[i] = 1
        else:
            y[i] = -1
    return y

def AnisotropicTV_image(y, y0, lam):

    d, d = np.shape(y)

    D = DIFF(d)

    et = np.ones(d-1)
    c = np.abs(1-y) - np.abs(y)

    # Construct the problem.
    x11 = cp.Variable((d-1,d))
    x12 = cp.Variable((d,d-1))
    x21 = cp.Variable((d-1,d))
    x22 = cp.Variable((d,d-1))
    x = cp.Variable((d,d))

    objective = cp.Minimize(cp.sum(x11) + cp.sum(x21) + cp.sum(x12) + cp.sum(x22) + lam * cp.sum(cp.multiply(c , x)))
    constraints = [ 0 <= x11, 0 <= x21, 0 <= x21, 0 <= x22, D @ x == x11 - x21, x @ D.T == x12 - x22 , x <= 1, 0 <= x]

    prob = cp.Problem(objective, constraints)

    #result = prob.solve(verbose=True)
    result = prob.solve()

    xsol = x.value
    #x1sol = x1.value
    #x2sol = x2.value

    # projection onto S_0 
    for i in range(d):
        for j in range(d):
            if xsol[i,j]> 0.5:
                xsol[i,j] = 1
            else:
                xsol[i,j] = 0

    s = MIoU(2*y0-1, 2*xsol-1)
    s = np.nanmean(s)

    if np.linalg.norm(np.abs(2*xsol-1) - np.ones(np.shape(xsol))) > 10**(-5):
        print('not tight')

    datei = open('data_0D_2Dgrid_aniso_tv.txt','a')
    datei.write(str(np.sqrt(4*np.sum((xsol - y0)**2))))
    datei.write('\n')
    datei.write(str(np.sum(2*np.abs(xsol - y0))))
    datei.write('\n')
    datei.write(str(s))
    datei.write('\n')

    print('RMSE \t\t| MAE \t\t| MioU')
    print('--------------------------------------------------')
    print("%10.2e"% (4*np.sum((xsol - y0)**2)), ' \t| ' , "%10.2e"% (2*np.sum(np.abs(xsol - y0))), ' \t|', "%10.2e"% s)


    return [result, xsol]

def mask(n,scale,q_tv_box):
    F = np.zeros(np.shape(q_tv_box))

    for i in range(n):
        for j in range(n):
            s = np.sum(q_tv_box[scale*i:scale*i+scale,scale*j:scale*j+scale])
            if s < 0:
                for k in range(scale):
                    for l in range(scale):
                        if np.round(q_tv_box[scale*i+k, scale*j+l]) == -1:
                            F[scale*i+k, scale*j+l] = 1
            if s > 0:
                for k in range(scale):
                    for l in range(scale):
                        if np.round(q_tv_box[scale*i+k, scale*j+l]) == 1:
                            F[scale*i+k, scale*j+l] = 1

    return F

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

def ISTA(n, scale, F, q_tv_box, dd_0):
    x = np.zeros(np.shape(q_tv_box))

    lam = 0.2

    m,m = np.shape(q_tv_box)

    x1 = np.random.randn(m,m)

    k = 0

    data = np.random.rand(0)

    for k in range(200000):
    #while k == 0 or data[k-1] - data[k] > 10**(-10):
        x1 = x
        s = F*q_tv_box
        for i in range(n):
            for j in range(n):
                x[scale*i:scale*i+scale,scale*j:scale*j+scale] = s[scale*i:scale*i+scale,scale*j:scale*j+scale] + (1 - F[scale*i:scale*i+scale,scale*j:scale*j+scale])*(x[scale*i:scale*i+scale,scale*j:scale*j+scale] - lam*grad(x[scale*i:scale*i+scale,scale*j:scale*j+scale]))
        #x = s + (1 - F)*(x - lam*grad(x))

        f =  np.linalg.norm(x - dd_0)

        data = np.append(data, f)

        print('iteration', k, data[k])

        if k > 0 and np.abs(data[k-1] - data[k]) < 10**(-10):
            break

        k = k+1
    
    return x

####### PLOTS #######

def bar_code_plt(x, y, z, tv, k):

    pixel_per_bar = 4
    dpi = 100

    fig = plt.figure(figsize=(4*len(x) * pixel_per_bar / dpi, k), dpi=4*dpi)

    ax = fig.add_subplot(1, 4, 1)
    ax.set_axis_off()
    ax.imshow(x.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_axis_off()
    ax2.imshow(y.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_axis_off()
    ax3.imshow(z.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')

    ax3 = fig.add_subplot(1, 4, 4)
    ax3.set_axis_off()
    ax3.imshow(tv.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')
    #plt.savefig('S0_case_example.png',dpi=400)

    plt.show()

def qr_code_plt(x,y,z):
    fig = plt.figure(0,figsize=(20,5), dpi = 4*100)
    plt.rc('font', size=7.5) 

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(x, cmap='Greys')

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(y, cmap='Greys')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.imshow(np.sign(z), cmap='Greys')

    ax3 = fig.add_subplot(1, 4, 4)
    ax3.imshow(np.abs(x - np.sign(z)), cmap='Greys')
    #plt.savefig("0D_2D_solution_2.pdf",dpi=400)

    plt.show()

def qr_code_improve_ista_plt(x,y,z,w):
    fig = plt.figure(0,figsize=(20,5), dpi = 3*100)
    plt.rc('font', size=7.5) 

    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(x, cmap='Greys')

    ax1 = fig.add_subplot(1, 4, 2)
    ax1.imshow(y, cmap='Greys')

    ax2 = fig.add_subplot(1, 4, 3)
    ax2.imshow(z, cmap='Greys')

    ax3 = fig.add_subplot(1, 4, 4)
    ax3.imshow(w, cmap='Greys')
    #plt.savefig("0D_2D_solution_2_plus_ista.pdf",dpi=400)

    plt.show()