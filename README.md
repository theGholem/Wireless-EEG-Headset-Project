# Wireless EEG Headset Project

## Overview
This project designs and implements a wireless EEG headset system for real-time brainwave acquisition, filtering, analysis (FFT, PSD, Power Bands), and visualization (2D and 3D).

- **`function.py`**: Core EEG processing functions (connection, filtering, FFT, PSD, band powers).
- **`main.py`**: PyQt5-based graphical interface for user interaction and real-time visualization.

## Features
- Connects to OpenBCI Cyton board
- Real-time EEG acquisition and streaming
- Signal filtering (bandpass and 60Hz notch)
- Time domain and frequency domain visualization
- 3D EEG signal display
- Power per frequency band (Delta, Theta, Alpha, Beta, Gamma)
- Data recording to CSV

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/theGholem/Wireless-EEG-Headset-Project
cd Wireless-EEG-Headset-Project
```

2. **Create and activate a virtual environment**:
```bash
python -m venv EEG_GUI_env
# Activate the environment
source EEG_GUI_env/bin/activate  # Linux/Mac
.\EEG_GUI_env\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
python main.py
```

## Requirements
- Python 3.8+
- BrainFlow
- PyQt5
- pyqtgraph
- NumPy
- SciPy

## Notes
- Verify your OpenBCI board connection and correct COM port before launching.
- Data files are saved by default to `C:\Users\Konan\Desktop\EEG_GUI`. Update in `main.py` if needed.

## Project Structure
```
├── function.py       # Core EEG functions
├── main.py           # GUI application
├── requirements.txt  # Project dependencies
├── README.md         # Documentation
└── test/             # Test and quality reports
```
