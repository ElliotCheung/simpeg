import numpy as np
from scipy.constants import mu_0, epsilon_0
from scipy.special import j0, j1, jn_zeros

from ...utils import omega
import importlib.resources

coef_path = importlib.resources.path('SimPEG.electromagnetics.wireless_electro_magnetic.utils', 
    'coefficients.npz')

def getEHfields(mesh1d, sigma1d, freq, zd=None, h_0=0, I=1, DL=1, fi=0, r=1e6, qwe_order=0, key=False):
    """
    Calculate the recieved field response

    """
    if mesh1d.dim != 1:
        raise NotImplementedError("Currently the WEM module can only deal with 1d model.")

    if zd is None:
        zd = np.array([0])

    # Setting up some frequently used constants.
    PE = I * DL
    cofi = np.cos(fi/180*np.pi)
    coefs = np.load(coef_path)
    wj0 = coefs['wj0']
    wj1 = coefs['wj1']
    if not qwe_order:
        ybase = coefs['YBASE']
        m = ybase / r
    else:
        temp, _ = np.polynomial.legendre.leggauss(200)
        temp+=1
        if key:
            abscissae = np.sort(np.hstack([0, jn_zeros(0, qwe_order), jn_zeros(1, qwe_order)]))
            m = np.hstack([abscissae[i]+temp*0.5*(abscissae[i+1]-abscissae[i]) for i in range(2*qwe_order)]) / r
        else:
            m = np.hstack([.65*(3**((i/qwe_order)**3)*temp+np.sum([2*3**((j/qwe_order)**3) for j in range(i)])) for i in range(qwe_order)]) / r

    k = np.sqrt(-1j * omega(freq) * mu_0 * sigma1d - omega(freq)**2 * epsilon_0 * mu_0)
    u = np.ones((len(sigma1d),len(m)), dtype=np.complex128)
    R1 = np.ones((len(sigma1d),len(m)), dtype=np.complex128)
    R2 = np.ones((len(sigma1d),len(m)), dtype=np.complex128)
    
    u = [np.sqrt(m**2 + k[i]**2) for i in range(len(sigma1d))]
    for i in range(len(sigma1d)-1):
        R1[i+1] = 1. / np.tanh(u[i+1] * mesh1d.edge_x_lengths[i+1] + np.arctanh(u[i]/u[i+1]/R1[i]))
        R2[i+1] = 1. / np.tanh(u[i+1] * mesh1d.edge_x_lengths[i+1] + np.arctanh(sigma1d[i+1]*u[i] /sigma1d[i]/u[i+1]/R2[i]))

    # create empty output fields with input size
    Ex = np.empty((len(zd),), dtype=np.complex128)
    Hy = np.empty((len(zd),), dtype=np.complex128)

    iind = np.nonzero(mesh1d.nodes_x[-2] < zd)[0]                                      # indices of ionosphere block
    aind = np.nonzero((mesh1d.nodes_x[-2] >= zd)*(mesh1d.nodes_x[-3] < zd))[0]         # indices of air block

    c01, d01, c02, d02, c1, d1, c2, d2 = return_variables(mesh1d, h_0, m, k, u, R1, R2)

    # Calculate field value of ionosphere
    for i in iind:
        ind = (np.abs(u[-2]*zd[i])>10)
        d01[ind]=0
        d02[ind]=0

        c11 = np.exp(u[-2]*mesh1d.edge_x_lengths[-2] + np.log(d01+c01*np.exp(-2*u[-2]*mesh1d.edge_x_lengths[-2])+
            m/u[-2]*np.exp(-u[-2]*np.abs(-mesh1d.edge_x_lengths[-2]+h_0)-u[-2]*mesh1d.edge_x_lengths[-2]))+u[-1]*mesh1d.edge_x_lengths[-2])
        c12 = np.exp(u[-2]*mesh1d.edge_x_lengths[-2] + np.log(d02+c02*np.exp(-2*u[-2]*mesh1d.edge_x_lengths[-2]))+u[-1]*mesh1d.edge_x_lengths[-2])

        X = np.exp(-u[-1]*zd[i]+np.log(c11))
        Z = np.exp(-u[-1]*zd[i]+np.log(c12))
        Z_1 = u[-1]*Z
        X_1 = u[-1]*X
        V = Z + X_1/m**2
        V_1 = Z_1 + u[-1]**2 * X / m**2
        Ex[i], Hy[i] = calculate_EH(m, wj0, wj1, r, PE, freq, cofi, k[-1], X, X_1, V, V_1, qwe_order, key)

    # Calculate field value in the air
    for i in aind:
        X = np.exp(u[-2]*zd[i] + np.log(d01+c01*np.exp(-2*u[-2]*zd[i]) + m/u[-2]*np.exp(-u[-2]*np.abs(-zd[i]+h_0)-u[-2]*zd[i])))
        X_1 = np.exp(u[-2]*zd[i] + np.log(-d01+c01*np.exp(-2*u[-2]*zd[i]) + m/u[-2]*np.exp(-u[-2]*np.abs(-zd[i]+h_0)-u[-2]*zd[i])) + np.log(u[-2]))
        Z = np.exp(u[-2]*zd[i] + np.log(d02+c02*np.exp(-2*u[-2]*zd[i])))
        Z_1 = np.exp(u[-2]*zd[i] + np.log(u[-2]) + np.log(-d02+c02*np.exp(-2*u[-2]*mesh1d.cell_centers_x[-2])))
        V = Z + X_1/m**2
        V_1 = Z_1 + u[-2]**2 * X/m**2
        Ex[i], Hy[i] = calculate_EH(m, wj0, wj1, r, PE, freq, cofi, k[-2], X, X_1, V, V_1, qwe_order, key)

    # Calculate field value underneath
    T111 = np.exp(np.log(d1)-u[-3]*mesh1d.edge_x_lengths[-3])
    T112 = np.exp(np.log(c1)+u[-3]*mesh1d.edge_x_lengths[-3])
    T121 = np.exp(np.log(d2)-u[-3]*mesh1d.edge_x_lengths[-3])
    T122 = np.exp(np.log(c2)+u[-3]*mesh1d.edge_x_lengths[-3])

    ind = (np.abs(T111)<np.abs(T112))
    T112[ind] = 0

    ind = (np.abs(T121)<np.abs(T122))
    T122[ind] = 0

    X = T111 + T112
    X_1 = -u[-3]*T111 + u[-3]*T112
    V = T121 + T122
    V_1 = -u[-3]*T121 + u[-3]*T122
    isthatlayer = np.nonzero((zd>mesh1d.nodes_x[-4])*(zd<=mesh1d.nodes_x[-3]))[0] # indices of current earth block
    for i in isthatlayer:
        F1, F2, F3, F4 = return_variable_(m, k[-3], u[-3], zd[i], c1, d1, c2, d2)
        Ex[i], Hy[i] = calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k[-3], F1, F2, F3, F4, qwe_order, key)

    for i in range(mesh1d.n_edges_x-4):
        c1 = np.exp(np.log(u[-i-4]*X + X_1) - np.log(2*u[-i-4])+u[-i-4]*mesh1d.nodes_x[-i-4])
        d1 = np.exp(np.log(u[-i-4]*X - X_1) - np.log(2*u[-i-4])-u[-i-4]*mesh1d.nodes_x[-i-4])
        c2 = np.exp(np.log(u[-i-4]*V + (k[-i-4]/k[-i-3])**2 * V_1) - np.log(2*u[-i-4]) + u[-i-4]*mesh1d.nodes_x[-i-4])
        d2 = np.exp(np.log(u[-i-4]*V - (k[-i-4]/k[-i-3])**2 * V_1) - np.log(2*u[-i-4]) - u[-i-4]*mesh1d.nodes_x[-i-4])

        T211 = np.exp(np.log(d1)+u[-i-4]*mesh1d.nodes_x[-i-5])
        T212 = np.exp(np.log(c1)-u[-i-4]*mesh1d.nodes_x[-i-5])
        T221 = np.exp(np.log(d2)+u[-i-4]*mesh1d.nodes_x[-i-5])
        T222 = np.exp(np.log(c2)-u[-i-4]*mesh1d.nodes_x[-i-5])

        ind = (np.abs(T211)<np.abs(T212))
        T212[ind] = 0
        ind = (np.abs(T221)<np.abs(T222))
        T222[ind] = 0
        
        X = T211 + T212
        X_1 = -u[-i-4]*T211 + u[-i-4]*T212
        V = T221 + T222
        V_1 = -u[-i-4]*T221 + u[-i-4]*T222
        isthatlayer = np.nonzero((zd>mesh1d.nodes_x[-i-5])*(zd<=mesh1d.nodes_x[-i-4]))[0] # indices of current earth block
        for i in isthatlayer:
            F1, F2, F3, F4 = return_variable_(m, k[-i-4], u[-i-4], zd[i], c1, d1, c2, d2)
            Ex[i], Hy[i] = calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k[-i-4], F1, F2, F3, F4, qwe_order, key)

    c1 = np.zeros((len(c1)))
    d1 = np.exp(np.log(X) - u[0]*mesh1d.nodes_x[1])
    c2 = np.zeros((len(c2)))
    d2 = np.exp(np.log(V) - u[0]*mesh1d.nodes_x[1])

    isthatlayer = np.nonzero(zd<=mesh1d.nodes_x[2])[0] # indices of current earth block
    for i in isthatlayer:
        T311 = np.exp(np.log(d1)+u[0]*zd[i])
        T312 = np.exp(np.log(c1)-u[0]*zd[i])
        T321 = np.exp(np.log(d2)+u[0]*zd[i])
        T322 = np.exp(np.log(c2)-u[0]*zd[i])
    
        ind = (np.abs(T311)<np.abs(T312))
        T312[ind] = 0
        ind = (np.abs(T321)<np.abs(T322))
        T322[ind] = 0

        F = d1*np.exp(u[0]*zd[i])
        FF = -u[0]*d2*np.exp(u[0]*zd[i])
        F1 = F
        F2 = FF - (k[0]/m)**2 * F
        F3 = -u[0] * d1 * np.exp(u[0]*zd[i])
        F4 = d2 * np.exp(u[0]*zd[i]) - F3/m**2
        Ex[i], Hy[i] = calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k[0], F1, F2, F3, F4, qwe_order, key)

    return Ex, np.zeros_like(Ex), Hy, np.zeros_like(Hy)

def return_variables(mesh1d, h_0, m, k, u, R1, R2):
    c0c1 = 1. / 2 * (1 + u[-1] / u[-2]) * np.exp((u[-1]-u[-2]) * -mesh1d.edge_x_lengths[-2])
    c0d1 = 1. / 2 * (1 - u[-1] / u[-2]) * np.exp((u[-1]+u[-2]) * -mesh1d.edge_x_lengths[-2])
    c0c2 = 1. / 2 * (1 + u[-1] / u[-2] * k[-2]**2 / k[-1]**2) * np.exp((u[-1]-u[-2]) * -mesh1d.edge_x_lengths[-2])
    c0d2 = 1. / 2 * (1 - u[-1] / u[-2] * k[-2]**2 / k[-1]**2) * np.exp((u[-1]+u[-2]) * -mesh1d.edge_x_lengths[-2])

    tmp = []
    tmp.append(c0c1 + c0d1)
    tmp.append(c0c1 - c0d1)
    tmp.append(c0c2 + c0d2)
    tmp.append(c0c2 - c0d2)
    tmp.append(tmp[1]/tmp[0])
    tmp.append(tmp[2]/tmp[3])
    tmp.append(u[-3]/R1[-3])
    tmp.append(k[-2]**2 / k[-3]**2 / u[-2] * tmp[5] + R2[-3] / u[-3])

    XX = m / u[-2] * (np.exp(-u[-2]*h_0) - np.exp(u[-2]*h_0)) + (-m / u[-2] * (np.exp(-u[-2]*h_0) - np.exp(u[-2]*h_0)) * tmp[6] \
        + m * (np.exp(-u[-2]*h_0) + np.exp(u[-2]*h_0))) / (u[-2]*tmp[4] + tmp[6])
    XX_1 = -tmp[6] * XX
    VV_1 = (np.exp(-u[-2]*h_0)*(1+tmp[4]) + np.exp(u[-2]*h_0)*(1-tmp[4])) / m / tmp[7]
    VV = -R2[-3] / u[-3] * VV_1
    ZZ = VV - XX_1/m**2
    ZZ_1 = VV_1 - u[-3]**2 * XX / m**2
    
    c01 = (u[-2]*XX + XX_1) / 2. / u[-2]
    d01 = (u[-2]*XX - XX_1 - 2.*m) / 2. / u[-2]
    c02 = (u[-2]*ZZ + (k[-2]/k[-3])**2 * (XX+ZZ_1) - XX) / 2. / u[-2]
    d02 = (u[-2]*ZZ - (k[-2]/k[-3])**2 * (XX+ZZ_1) + XX) / 2. / u[-2]

    c1 = np.exp(np.log(u[-3]*XX + XX_1) - np.log(2*u[-3]))
    d1 = np.exp(np.log(u[-3]*XX - XX_1) - np.log(2*u[-3]))
    c2 = np.exp(np.log(u[-3]*VV + VV_1) - np.log(2*u[-3]))
    d2 = np.exp(np.log(u[-3]*VV - VV_1) - np.log(2*u[-3]))

    return c01,d01,c02,d02,c1,d1,c2,d2

    
def calculate_EH(m, wj0, wj1, r, PE, freq, cofi, k, X, X_1, V, V_1, qwe_order, key):
    F=X
    FF = V_1
    F1 = F
    F2 = FF - (k/m)**2 * F
    F3 = X_1
    F4 = V - X_1 / m**2

    if not qwe_order:
        I1 = np.sum(F1 * wj0) / r
        I2 = np.sum(F2 * m * wj1) / r
        I3 = np.sum(F2 * m**2 * wj0) / r

        I4 = np.sum(F3 * wj0) / r
        I5 = np.sum(F4 * m * wj1) / r
        I6 = np.sum(F4 * m**2 * wj0) /r
    else:
        I1 = EpsShanks(F1 * j0(m*r), qwe_order, key) / r
        I2 = EpsShanks(F2 * m * j1(m*r), qwe_order, key) / r
        I3 = EpsShanks(F2 * m**2 * j0(m*r), qwe_order, key) / r
        
        I4 = EpsShanks(F3 * j0(m*r), qwe_order, key) / r
        I5 = EpsShanks(F4 * m * j1(m*r), qwe_order, key) / r
        I6 = EpsShanks(F4 * m**2 * j0(m*r), qwe_order, key) / r

    Ex = PE / 4 / np.pi * 1j * omega(freq) * mu_0 * \
        (I1 + 1 / k**2 / r * (1 - 2 * cofi**2)*I2 + 1 / k**2 * cofi**2 * I3)
    
    Hy = PE / 4 / np.pi * (I4 + 1 / r * (1 - 2 * cofi**2) * I5 + I6 * cofi**2)
    return Ex, Hy

def calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k, F1, F2, F3, F4, qwe_order, key):
    if not qwe_order:
        I1 = np.sum(F1 * wj0) / r
        I2 = np.sum(F2 * m * wj1) / r
        I3 = np.sum(F2 * m**2 * wj0) / r

        I4 = np.sum(F3 * wj0) / r
        I5 = np.sum(F4 * m * wj1) / r
        I6 = np.sum(F4 * m**2 * wj0) /r
    else:
        I1 = EpsShanks(F1 * j0(m*r), qwe_order, key) / r
        I2 = EpsShanks(F2 * m * j1(m*r), qwe_order, key) / r
        I3 = EpsShanks(F2 * m**2 * j0(m*r), qwe_order, key) / r
        
        I4 = EpsShanks(F3 * j0(m*r), qwe_order, key) / r
        I5 = EpsShanks(F4 * m * j1(m*r), qwe_order, key) / r
        I6 = EpsShanks(F4 * m**2 * j0(m*r), qwe_order, key) / r

    Ex = PE / 4 / np.pi * 1j * omega(freq) * mu_0 * \
        (I1 + 1 / k**2 / r * (1 - 2 * cofi**2)*I2 + 1 / k**2 * cofi**2 * I3)
    
    Hy = PE / 4 / np.pi * (I4 + 1 / r * (1 - 2 * cofi**2) * I5 + I6 * cofi**2)
    return Ex, Hy

def EpsShanks(arr_, order, key, trim_a=1e-15, trim_b=1e-38):
    if key:
        num_pack = int(len(arr_) / order / 2)
        _, w0 = np.polynomial.legendre.leggauss(num_pack)
        abscissae = np.sort(np.hstack([0, jn_zeros(0, order), jn_zeros(1, order)]))
        arr = np.array([.5*(abscissae[i+1]-abscissae[i])*np.sum(arr_[i*num_pack:(i+1)*num_pack]*w0) for i in range(2*order)])
    else:
        num_pack = int(len(arr_) / order)
        _, w0 = np.polynomial.legendre.leggauss(num_pack)
        arr = np.hstack([.65*3**((i/order)**3)*np.sum(arr_[i*num_pack:(i+1)*num_pack]*w0) for i in range(order)])
    indx = 0
    currentSum = arr[indx]
    result = currentSum
    result_prev = 0
    err = np.abs(result - result_prev)
    temp = [np.array([currentSum], dtype=np.complex128)]

    while err>(trim_a*np.abs(result)+trim_b) and indx<len(arr)-2:
        indx += 1
        temp.append(np.zeros(indx+1, dtype=np.complex128))
        F = arr[indx]
        currentSum += F
        temp[-1][0] = currentSum
        temp[-1][1] = 1. / F
        if indx>1:
            for i in range(len(temp[-1])-2):
                #with np.errstate(divide="raise"):
                temp[-1][i+2] = temp[-2][i] + 1. / (temp[-1][i+1] - temp[-2][i+1])
        
        result_prev = result
        if indx%2==0:
            result = temp[-1][-1]
        else:
            result = temp[-1][-2]
        
        err = np.abs(result - result_prev)
        temp.remove(temp[0])

    return result

def return_variable_(m, k, u, zd, c1, d1, c2, d2):
    T311 = np.exp(np.log(d1)+u*zd)
    T312 = np.exp(np.log(c1)-u*zd)
    T321 = np.exp(np.log(d2)+u*zd)
    T322 = np.exp(np.log(c2)-u*zd)

    ind = (np.abs(T311)<np.abs(T312))
    T312[ind] = 0
    ind = (np.abs(T321)<np.abs(T322))
    T322[ind] = 0

    F = T311 + T312
    FF = -u*T321 + u*T322
    F1 = F
    F2 = FF - (k/m)**2 * F
    F3 = -u*T311 + u*T312
    F4 = T321 + T322 - F3/m**2
    return F1,F2,F3,F4