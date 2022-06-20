import numpy as np
from scipy.constants import mu_0, epsilon_0

from ...utils import omega
import importlib.resources

coef_path = importlib.resources.path('SimPEG.electromagnetics.wireless_electro_magnetic.utils', 
    'coefficients.npz')

def getEHfields(mesh1d, sigma1d, freq, h_0=0, I=1, DL=1, fi=0, r=1e6):
    """
    Calculate the recieved field response

    """
    if mesh1d.dim != 1:
        raise NotImplementedError("Currently the WEM module can only deal with 1d model.")

    # Setting up some frequently used constants.
    PE = I * DL
    cofi = np.cos(fi/180*np.pi)
    coefs = np.load(coef_path)
    wj0 = coefs['wj0']
    wj1 = coefs['wj1']
    ybase = coefs['YBASE']
    
    k = np.sqrt(-1j * omega(freq) * mu_0 * sigma1d - omega(freq)**2 * epsilon_0 * mu_0)
    u = np.ones((len(sigma1d),len(wj0)), dtype=np.complex128)
    R1 = np.ones((len(sigma1d),len(wj0)), dtype=np.complex128)
    R2 = np.ones((len(sigma1d),len(wj0)), dtype=np.complex128)
    m = ybase / r
    
    u = [np.sqrt(m**2 + k[i]**2) for i in range(len(sigma1d))]
    for i in range(len(sigma1d)-1):
        R1[i+1] = 1. / np.tanh(u[i+1] * mesh1d.edge_x_lengths[i+1] + np.arctanh(u[i]/u[i+1]/R1[i]))
        R2[i+1] = 1. / np.tanh(u[i+1] * mesh1d.edge_x_lengths[i+1] + np.arctanh(sigma1d[i+1]*u[i] /sigma1d[i]/u[i+1]/R2[i]))

    c0c1 = 1. / 2 * (1 + u[-2] / u[-1]) * np.exp((u[-2]-u[-1]) * -mesh1d.edge_x_lengths[1])
    c0d1 = 1. / 2 * (1 - u[-2] / u[-1]) * np.exp((u[-2]+u[-1]) * -mesh1d.edge_x_lengths[1])
    c0c2 = 1. / 2 * (1 + u[-2] / u[-1] * k[-1]**2 / k[-2]**2) * np.exp((u[-2]-u[-1]) * -mesh1d.edge_x_lengths[1])
    c0d2 = 1. / 2 * (1 - u[-2] / u[-1] * k[-1]**2 / k[-2]**2) * np.exp((u[-2]+u[-1]) * -mesh1d.edge_x_lengths[1])

    tmp = []
    tmp.append(c0c1 + c0d1)
    tmp.append(c0c1 - c0d1)
    tmp.append(c0c2 + c0d2)
    tmp.append(c0c2 - c0d2)
    tmp.append(tmp[1]/tmp[0])
    tmp.append(tmp[2]/tmp[3])
    tmp.append(u[-3]/R1[-3])
    tmp.append(k[-1]**2 / k[-3]**2 / u[-1] * tmp[5] + R2[-3] / u[-3])

    XX = m / u[-1] * (np.exp(-u[-1]*h_0) - np.exp(u[-1]*h_0)) * (1 - tmp[6]) + m * (np.exp(-u[-1]*h_0) + np.exp(u[-1]*h_0)) / (u[-1]*tmp[4] + tmp[6])
    XX_1 = -tmp[6] * XX
    VV_1 = (np.exp(-u[-1]*h_0)*(1+tmp[4]) + np.exp(u[-1]*h_0)*(1-tmp[4])) / m / tmp[7]
    VV = -R2[-3] / u[-3] * VV_1
    ZZ = VV - XX_1/m**2
    ZZ_1 = VV_1 - u[-3]**2 * XX / m**2
    
    c01 = (u[-1]*XX + XX_1) / 2. / u[-1]
    d01 = (u[-1]*XX - XX_1 - 2.*m) / 2. / u[-1]
    c02 = (u[-1]*ZZ + (k[-1]/k[-3])**2 * (XX+ZZ_1) - XX) / 2. / u[-1]
    d02 = (u[-1]*ZZ - (k[-1]/k[-3])**2 * (XX+ZZ_1) + XX) / 2. / u[-1]

    ind = (np.abs(u[-1]*mesh1d.cell_centers_x[-1])>10)
    d01[ind]=0
    d02[ind]=0

    Ex = np.empty((mesh1d.n_edges_x,), dtype=np.complex128)
    Hy = np.empty((mesh1d.n_edges_x,), dtype=np.complex128)

    # Calculate field value of ionosphere
    c11 = np.exp(u[-1]*mesh1d.edge_x_lengths[-2] + np.log(d01+c01*np.exp(-2*u[-1]*mesh1d.edge_x_lengths[-2])+
        m/u[-1]*np.exp(-u[-1]*np.abs(-mesh1d.edge_x_lengths[-2]+h_0)-u[-1]*mesh1d.edge_x_lengths[-2]))+u[-2]*mesh1d.edge_x_lengths[-2])
    c12 = np.exp(u[-1]*mesh1d.edge_x_lengths[-2] + np.log(d02+c02*np.exp(-2*u[-1]*mesh1d.edge_x_lengths[-2]))+u[-2]*mesh1d.edge_x_lengths[-2])

    X = np.exp(-u[-2]*mesh1d.cell_centers_x[-1]+np.log(c11))
    Z = np.exp(-u[-2]*mesh1d.cell_centers_x[-1]+np.log(c12))
    Z_1 = u[-2]*Z
    X_1 = u[-2]*X
    V = Z + X_1/m**2
    V_1 = Z_1 + u[-2]**2 * X / m**2
    Ex[-1], Hy[-1] = calculate_EH(m, wj0, wj1, r, PE, freq, cofi, k[-2], X, X_1, V, V_1)

    # Calculate field value in the air
    X = np.exp(u[-1]*mesh1d.cell_centers_x[-2] + np.log(d01+c01*np.exp(-2*u[-1]*mesh1d.cell_centers_x[-2]) + 
        m/u[-1]*np.exp(-u[-1]*np.abs(-mesh1d.cell_centers_x[-2]+h_0)-u[-1]*mesh1d.cell_centers_x[-2])))
    X_1 = np.exp(u[-1]*mesh1d.cell_centers_x[-2] + np.log(-d01+c01*np.exp(-2*u[-1]*mesh1d.cell_centers_x[-2]) + 
        m / u[-1]*np.exp(-u[-1]*np.abs(-mesh1d.cell_centers_x[-2]+h_0)-u[-1]*mesh1d.cell_centers_x[-2])) + np.log(u[-1]))
    Z = np.exp(u[-1]*mesh1d.cell_centers_x[-2] + np.log(d02+c02*np.exp(-2*u[-1]*mesh1d.cell_centers_x[-2])))
    Z_1 = np.exp(u[-1]*mesh1d.cell_centers_x[-2] + np.log(u[-1]) + np.log(-d02+c02*np.exp(-2*u[-1]*mesh1d.cell_centers_x[-2])))
    V = Z + X_1/m**2
    V_1 = Z_1 + u[-1]**2 * X/m**2
    Ex[-2], Hy[-2] = calculate_EH(m, wj0, wj1, r, PE, freq, cofi, k[0], X, X_1, V, V_1)

    # Calculate field value underneath
    c1 = np.exp(np.log(u[-3]*XX + XX_1) - np.log(2*u[-3]))
    d1 = np.exp(np.log(u[-3]*XX - XX_1) - np.log(2*u[-3]))
    c2 = np.exp(np.log(u[-3]*VV + VV_1) - np.log(2*u[-3]))
    d2 = np.exp(np.log(u[-3]*VV - VV_1) - np.log(2*u[-3]))
    
    T311 = np.exp(np.log(d1)+u[-3]*mesh1d.cell_centers_x[-3])
    T312 = np.exp(np.log(c1)-u[-3]*mesh1d.cell_centers_x[-3])
    T321 = np.exp(np.log(d2)+u[-3]*mesh1d.cell_centers_x[-3])
    T322 = np.exp(np.log(c2)-u[-3]*mesh1d.cell_centers_x[-3])

    ind = (np.abs(T311)<np.abs(T312))
    T312[ind] = 0
    ind = (np.abs(T321)<np.abs(T322))
    T322[ind] = 0

    F = T311 + T312
    FF = -u[-3]*T321 + u[-3]*T322
    F1 = F
    F2 = FF - (k[-3]/m)**2 * F
    F3 = -u[-3]*T311 + u[-3]*T312
    F4 = T321 + T322 - F3/m**2

    Ex[-3], Hy[-3] = calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k[-3], F1, F2, F3, F4)
    
    if mesh1d.n_edges_x>3:
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

            T311 = np.exp(np.log(d1)+u[-i-4]*mesh1d.cell_centers_x[-i-4])
            T312 = np.exp(np.log(c1)-u[-i-4]*mesh1d.cell_centers_x[-i-4])
            T321 = np.exp(np.log(d2)+u[-i-4]*mesh1d.cell_centers_x[-i-4])
            T322 = np.exp(np.log(c2)-u[-i-4]*mesh1d.cell_centers_x[-i-4])

            ind = (np.abs(T311)<np.abs(T312))
            T312[ind] = 0
            ind = (np.abs(T321)<np.abs(T322))
            T322[ind] = 0

            F = T311 + T312
            FF = -u[-i-4]*T321 + u[-i-4]*T322
            F1 = F
            F2 = FF - (k[-i-4]/m)**2 * F
            F3 = -u[-i-4]*T311 + u[-i-4]*T312
            F4 = T321 + T322 - F3/m**2
            Ex[-i-4], Hy[-i-4] = calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k[-i-4], F1, F2, F3, F4)

        c1 = np.zeros((len(c1)))
        d1 = np.exp(np.log(X) - u[0]*mesh1d.nodes_x[1])
        c2 = np.zeros((len(c2)))
        d2 = np.exp(np.log(V) - u[0]*mesh1d.nodes_x[1])

        T311 = np.exp(np.log(d1)+u[0]*mesh1d.cell_centers_x[0])
        T312 = np.exp(np.log(c1)-u[0]*mesh1d.cell_centers_x[0])
        T321 = np.exp(np.log(d2)+u[0]*mesh1d.cell_centers_x[0])
        T322 = np.exp(np.log(c2)-u[0]*mesh1d.cell_centers_x[0])
        
        ind = (np.abs(T311)<np.abs(T312))
        T312[ind] = 0
        ind = (np.abs(T321)<np.abs(T322))
        T322[ind] = 0

        F = d1*np.exp(u[0]*mesh1d.cell_centers_x[0])
        FF = -u[0]*d2*np.exp(u[0]*mesh1d.faces_x[0])
        F1 = F
        F2 = FF - (k[0]/m)**2 * F
        F3 = -u[0] * d1 * np.exp(u[0]*mesh1d.cell_centers_x[0])
        F4 = d2 * np.exp(u[0]*mesh1d.cell_centers_x[0]) - F3/m**2
        Ex[0], Hy[0] = calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k[0], F1, F2, F3, F4)

        return Ex, np.zeros_like(Ex), Hy, np.zeros_like(Hy)

    
def calculate_EH(m, wj0, wj1, r, PE, freq, cofi, k, X, X_1, V, V_1):
    F=X
    FF = V_1
    F1 = F
    F2 = FF - (k/m)**2 * F
    F3 = X_1
    F4 = V - X_1 / m**2

    I1 = np.sum(F1 * wj0) / r
    I2 = np.sum(F2 * m * wj1) / r
    I3 = np.sum(F2 * m**2 * wj0) / r

    I4 = np.sum(F3 * wj0) / r
    I5 = np.sum(F4 * m * wj1) / r
    I6 = np.sum(F4 * m**2 * wj0) /r

    Ex = PE / 4 / np.pi * 1j * omega(freq) * mu_0 * \
        (I1 + 1 / k**2 / r * (1 - 2 * cofi**2)*I2 + 1 / k**2 * cofi**2 * I3)
    
    Hy = PE / 4 / np.pi * (I4 + 1 / r * (1 - 2 * cofi**2) * I5 + I6 * cofi**2)
    return Ex, Hy

def calculate_EH_(m, wj0, wj1, r, PE, freq, cofi, k, F1, F2, F3, F4):
    I1 = np.sum(F1 * wj0) / r
    I2 = np.sum(F2 * m * wj1) / r
    I3 = np.sum(F2 * m**2 * wj0) / r

    I4 = np.sum(F3 * wj0) / r
    I5 = np.sum(F4 * m * wj1) / r
    I6 = np.sum(F4 * m**2 * wj0) /r

    Ex = PE / 4 / np.pi * 1j * omega(freq) * mu_0 * \
        (I1 + 1 / k**2 / r * (1 - 2 * cofi**2)*I2 + 1 / k**2 * cofi**2 * I3)
    
    Hy = PE / 4 / np.pi * (I4 + 1 / r * (1 - 2 * cofi**2) * I5 + I6 * cofi**2)
    return Ex, Hy