from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import welch, filtfilt, cheby1
import numpy as np

def prepare_board(com_port):
    """
    Prépare la connexion à la carte OpenBCI Cyton.
    """
    # Initialiser les paramètres de la carte (port série, etc.)
    params = BrainFlowInputParams()
    params.serial_port = com_port  # Remplacez 'COM3' par votre port COM réel

    # Créer un objet Board pour la carte Cyton
    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)

    # Préparer la carte pour l'acquisition de données
    BoardShim.enable_dev_board_logger()  # Activer le logger de la carte (debug)
    try:
        board.prepare_session()  # Prépare la session d'acquisition
        status = 'Board is ready and connected!'
    except Exception as e:
        print('Error: ', e)
        status = e
    return board, board_id, status

def compute_fft_welch(eeg_data, fs, nperseg=None):
    """
    Calcule la FFT (méthode de Welch) pour les données EEG fournies.
    """
    # Ajuster nperseg pour qu'il ne dépasse pas la longueur des données
    if nperseg is None or nperseg > eeg_data.shape[1]:
        nperseg = min(fs, eeg_data.shape[1])
    # Pré-allouer le tableau PSD (chaque canal aura nperseg//2+1 points de fréquence)
    psds = np.empty((eeg_data.shape[0], nperseg//2 + 1))
    valid_channels = [i for i, ch_data in enumerate(eeg_data) if len(ch_data) >= nperseg]

    # Calculer la densité spectrale de puissance pour chaque canal valide
    for i in valid_channels:
        _, psd = welch(eeg_data[i], fs=fs, nperseg=nperseg)
        psds[i] = psd

    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    return freqs, psds

def compute_power_bands(freqs, psds):
    """
    Calcule la puissance par bande de fréquences (Delta, Theta, Alpha, Beta, Gamma).
    """
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    # Indices de fréquence pour chaque bande
    band_indices = {
        band: np.logical_and(freqs >= low, freqs <= high)
        for band, (low, high) in bands.items()
    }
    psd_mean = np.mean(psds, axis=0)
    # Intégrer la PSD sur chaque bande de fréquence
    band_powers = {
        band: np.trapz(psd_mean[idx], freqs[idx]) for band, idx in band_indices.items()
    }
    return np.array(list(band_powers.values()))

def cheby1_bandpass(lowcut, highcut, fs, order=4, rp=0.5):
    """
    Conçoit un filtre passe-bande Chebyshev de type I.
    """
    nyq = 0.5 * fs  # Fréquence de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, rp, [low, high], btype='band')
    return b, a

def cheby1_notch(fs, center_freq=60, band_width=1, order=4, rp=0.5):
    """
    Conçoit un filtre coupe-bande (notch) Chebyshev de type I.
    """
    nyq = 0.5 * fs  # Fréquence de Nyquist
    low = (center_freq - band_width / 2) / nyq
    high = (center_freq + band_width / 2) / nyq
    b, a = cheby1(order, rp, [low, high], btype='bandstop')
    return b, a

def eeg_filtering(eeg_data, fs, lowcut=0.1, highcut=100, order=4, rp=0.5):
    """
    Filtre passe-bande + notch 60 Hz sur les données EEG de chaque canal.
    """
    # Conception des filtres passe-bande et notch
    b, a = cheby1_bandpass(lowcut, highcut, fs, order, rp)
    b1, a1 = cheby1_notch(fs, 60, 1, order, rp)
    filtered_data = np.zeros_like(eeg_data)  # Tableau pour les données filtrées

    # Appliquer le filtre sur chaque canal EEG
    for i in range(eeg_data.shape[0]):
        # Retirer la composante DC (moyenne) du canal
        filtered_data[i] = eeg_data[i] - np.mean(eeg_data[i])
        # Filtre passe-bande
        filtered_data[i] = filtfilt(b, a, filtered_data[i], axis=-1)
        # Filtre notch 60 Hz
        filtered_data[i] = filtfilt(b1, a1, filtered_data[i], axis=-1)
    return filtered_data
