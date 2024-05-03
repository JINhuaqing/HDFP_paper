import numpy as np


_par_low = np.array([0.1, 0.001,0.001, 0.005, 0.005, 0.005, 5])
_par_high = np.asarray([1, 0.7, 2, 0.03, 0.03, 0.20, 20])
_prior_bds = np.array([_par_low, _par_high]).T
#names = ["alpha", "gei", "gii", "Taue", "TauG", "Taui", "Speed"]

def logistic_np(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    num = np.exp(k*x)
    den = np.exp(k*x) + 1
    # fix inf issue
    res = num/den
    res[np.isinf(num)] = 1
    return res

def raw2theta_np(thetas_raw, k=0.15):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x 7
              _prior_bds: an array with 7 x 2
    """
    assert _prior_bds.shape[0] == 7
    assert thetas_raw.shape[-1] == 7
    thetas = logistic_np(thetas_raw, k=k)*(_prior_bds[:, 1] -  _prior_bds[:, 0]) + _prior_bds[:, 0]
    return thetas


def network_transfer_local(C, D, parameters, w):
    """Network Transfer Function for spectral graph model for give freq w

    Args:
        brain (Brain): specific brain to calculate NTF
        parameters (dict): parameters for ntf. We shall keep this separate from Brain
        for now, as we want to change and update according to fitting.
        frequency (float): frequency at which to calculate NTF

    Returns:
        model_out (numpy asarray):  Each region's frequency response for
        the given frequency (w)
        frequency_response (numpy asarray):
        ev (numpy asarray): Eigen values
        Vv (numpy asarray): Eigen vectors

    """
    #C = brain.reducedConnectome
    #D = brain.distance_matrix

    
    parameters = np.asarray(parameters)
    alpha = parameters[0]
    gei =   parameters[1]  
    gii =   parameters[2]  
    tau_e = parameters[3]
    tauG =  parameters[4]
    tau_i = parameters[5]
    speed = parameters[6]
    gee = 1
    
    # Defining some other parameters used:
    zero_thr = 0.05

    # define sum of degrees for rows and columns for laplacian normalization
    rowdegree = np.transpose(np.sum(C, axis=1))
    coldegree = np.sum(C, axis=0)
    qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
    rowdegree[qind] = np.inf
    coldegree[qind] = np.inf

    nroi = C.shape[0]

    K = nroi

    Tau = 0.001 * D / speed
    Cc = C * np.exp(-1j * Tau * w)

    # Eigen Decomposition of Complex Laplacian Here
    L1 = np.identity(nroi)
    L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
    L = L1 - alpha * np.matmul(np.diag(L2), Cc)

    d, v = np.linalg.eig(L)  
    eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
    eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
    eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index

    eigenvalues = np.transpose(eig_val)
    eigenvectors = eig_vec[:, 0:K]

#     # Cortical model
    Fe = np.divide(1 / tau_e ** 2, (1j * w + 1 / tau_e) ** 2)
    Fi = np.divide(1 / tau_i ** 2, (1j * w + 1 / tau_i) ** 2)
    FG = np.divide(1 / tauG ** 2, (1j * w + 1 / tauG) ** 2)

    Hed = (1 + (Fe * Fi * gei)/(tau_e * (1j * w + Fi * gii/tau_i)))/(1j * w + Fe * gee/tau_e + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fi * gii / tau_i)))
    
    Hid = (1 - (Fe * Fi * gei)/(tau_i * (1j * w + Fe * gee/tau_e)))/(1j * w + Fi * gii/tau_i + (Fe * Fi * gei)**2/(tau_e * tau_i * (1j * w + Fe * gee / tau_e)))

    Htotal = Hed + Hid


#     q1 = (1j * w + 1 / tau_e * Fe * eigenvalues)
    q1 = (1j * w + 1 / tauG * FG * eigenvalues)
    qthr = zero_thr * np.abs(q1[:]).max()
    magq1 = np.maximum(np.abs(q1), qthr)
    angq1 = np.angle(q1)
    q1 = np.multiply(magq1, np.exp(1j * angq1))
    frequency_response = np.divide(Htotal, q1)
    
    model_out = 0

    for k in range(K):
        model_out += (frequency_response[k]) * np.outer(eigenvectors[:, k], np.conjugate(eigenvectors[:, k])) 
    model_out2 = np.linalg.norm(model_out,axis=1)

    
    return model_out2


class SGM:
    def __init__(self, C, D, freqs, k=0.15):
        self.freqs = freqs 
        self.C = C
        self.D = D
        self.k = k
        
    def __call__(self, raw_params, is_std=True):
        """run_forward. Function for running the forward model over the passed in range of frequencies,
        for the handed set of parameters (which must be passed in as a dictionary)
    
        Args:
            C = brain.reducedConnectome
            D = brain.distance_matrix
            params (dict): Dictionary of a setting of parameters for the NTF model.
            freqs (array): Array of freqencies for which the model is to be calculated, in Hz
            is_std: std the psd across freq axis or not, default is True
    
        Returns:
            model_out(array): PSD in dB, (maybe standardized)
    
        """
    
        params = raw2theta_np(raw_params, self.k)
        model_out = []
        
        for freq in self.freqs:
            w = 2 * np.pi * freq
            freq_model = network_transfer_local(
                self.C, self.D, params, w
            )
            model_out.append(freq_model)
        model_out = np.array(model_out).T
        
        model_out_dB = 20* np.log10(model_out)
        if is_std:
            model_out_dB = (model_out_dB-model_out_dB.mean(axis=-1, keepdims=True))/model_out_dB.std(axis=-1, keepdims=True)
        
        return model_out_dB
        
        
