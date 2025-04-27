from PyQt5 import QtCore, QtGui, QtWidgets
from function import *  # Importation des fonctions externes (prepare_board, eeg_filtering, etc.)
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import sys
import csv
import os
from datetime import datetime

class MainApp(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # Initialisation des variables d'état
        self.is_streaming = False
        self.record_data = False



    def reset_app(self):
        """Réinitialise tous les champs et arrête les processus en cours."""
        self.ui.com_port.clear()
        self.ui.win_size.clear()
        self.ui.fps.clear()
        self.ui.trial_name.clear()

        # Décocher toutes les cases à cocher
        for checkbox in [self.ui.BoxCh1, self.ui.BoxCh2, self.ui.BoxCh3, self.ui.BoxCh4,
                         self.ui.BoxCh5, self.ui.BoxCh6, self.ui.BoxCh7, self.ui.BoxCh8,
                         self.ui.BoxFiltering, self.ui.BoxFFT, self.ui.BoxPSD, self.ui.BoxTime]:
            checkbox.setChecked(False)

        # Effacer les graphiques
        self.ui.TimeGraph.clear()
        self.ui.FFTGraph.clear()
        self.ui.PSD.clear()

        # Arrêter le streaming s'il est actif
        if getattr(self, 'is_streaming', False):
            try:
                self.board.stop_stream()
            except Exception as e:
                print("Erreur lors de l'arrêt du stream:", e)
            self.is_streaming = False

        # Mettre à jour le label de statut
        self.ui.label_6.setText('Réinitialisation terminée.')



    def closeEvent(self, event):
        """Surcharge de l'événement de fermeture pour arrêter correctement les flux de données."""
        # Arrêter le streaming s'il est en cours
        if self.is_streaming:
            try:
                self.board.stop_stream()
            except Exception as e:
                print(f"Erreur lors de l'arrêt du flux de données : {e}")
        # Arrêter l'enregistrement s'il est en cours (et fermer les fichiers)
        if self.record_data:
            self.end_recording()
        event.accept()

    def connect_board(self):
        """Prépare la connexion à la carte EEG et démarre le streaming."""
        if not self.com_port.text() or not self.win_size.text() or not self.fps.text():
            self.label_6.setText("Veuillez renseigner le port, la fenêtre et le FPS.")
            return
        # Indiquer visuellement la tentative de connexion
        self.label_6.setText("Connecting to Board...")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        # Préparer la connexion à la carte (peut prendre quelques secondes)
        self.board, self.board_id, self.status = prepare_board(f'COM{self.com_port.text()}')
        QtWidgets.QApplication.restoreOverrideCursor()
        self.label_6.setText(self.status)
        if self.board is None:
            # Échec de connexion
            return
        # Démarrer le streaming EEG en cas de succès
        self.initialize_and_start_stream()

    def initialize_and_start_stream(self):
        """Démarre le streaming EEG et configure le timer de mise à jour."""
        self.is_streaming = True
        try:
            # Démarrer le flux de données EEG (buffer interne de 45000 points)
            self.board.start_stream(45000)
        except Exception as e:
            self.label_6.setText(f"Échec du démarrage du stream : {e}")
            return
        # Obtenir la fréquence d'échantillonnage et les indices de canaux EEG
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        # Configurer l'intervalle du timer en fonction du FPS souhaité
        try:
            base_interval = int(1000 / max(1, int(self.fps.text())))
        except Exception:
            base_interval = 1000
        self.timer.start(base_interval)
        self.label_6.setText("Connected and streaming data...")
        # Noms des bandes de fréquences pour l'affichage PSD
        self.bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    def update_data_and_graphs(self):
        """Récupère les nouvelles données EEG et met à jour les graphiques."""
        if not self.is_streaming:
            return
        # Obtenir les nouveaux échantillons du buffer interne de la carte
        data_chunk = self.board.get_board_data()
        if data_chunk.size == 0:
            return  # pas de nouvelles données pour l'instant
        eeg_data_chunk = data_chunk[self.eeg_channels, :]
        # Ajouter ces données au buffer global en conservant l'historique
        if hasattr(self, 'eeg_channel_data') and self.eeg_channel_data.size != 0:
            if self.eeg_channel_data.shape[0] == len(getattr(self, 'eeg_channel_indices', [])):
                self.eeg_channel_data = np.concatenate((self.eeg_channel_data, eeg_data_chunk), axis=1)
            else:
                self.eeg_channel_data = eeg_data_chunk
        else:
            self.eeg_channel_data = eeg_data_chunk
        # Mettre à jour la liste des canaux sélectionnés
        channel_boxes = [self.BoxCh1, self.BoxCh2, self.BoxCh3, self.BoxCh4,
                         self.BoxCh5, self.BoxCh6, self.BoxCh7, self.BoxCh8]
        selected = [cb.isChecked() for cb in channel_boxes]
        self.eeg_channel_indices = [i for i, sel in enumerate(selected) if sel]
        if not self.eeg_channel_indices:
            self.label_6.setText("Aucun canal sélectionné pour l'affichage.")
            return
        # Limiter la taille du buffer aux derniers N échantillons (fenêtre temporelle)
        max_samples = int(float(self.win_size.text()) * self.fs) if self.win_size.text() else self.eeg_channel_data.shape[1]
        if self.eeg_channel_data.shape[1] > max_samples:
            self.eeg_channel_data = self.eeg_channel_data[:, -max_samples:]
        # Filtrer le signal si l'option de filtrage est activée
        if self.BoxFiltering.isChecked():
            self.eeg_channel_data_filt = eeg_filtering(self.eeg_channel_data, self.fs)
        else:
            self.eeg_channel_data_filt = self.eeg_channel_data
        # Axe des temps (secondes) pour l'affichage temporel
        t = np.arange(self.eeg_channel_data_filt.shape[1]) / self.fs
        # Affichage dans le domaine temporel (si case cochée)
        if self.BoxTime.isChecked():
            self.TimeGraph.clear()
            for idx, ch_idx in enumerate(self.eeg_channel_indices):
                pen = pg.mkPen(['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange'][ch_idx % 8])
                self.TimeGraph.plot(t, self.eeg_channel_data_filt[idx, :] + idx * 2000, pen=pen)
        else:
            self.TimeGraph.clear()
        # Affichage du spectre (FFT) si demandé
        if self.BoxFFT.isChecked():
            freqs, psds = compute_fft_welch(self.eeg_channel_data_filt, self.fs)
            self.FFTGraph.clear()
            for idx, ch_idx in enumerate(self.eeg_channel_indices):
                pen = pg.mkPen(['r', 'g', 'b', 'c', 'm', 'y', 'w', 'orange'][ch_idx % 8])
                self.FFTGraph.plot(freqs, psds[idx, :], pen=pen)
        else:
            self.FFTGraph.clear()
        # Affichage de la puissance par bande (PSD) si demandé
        if self.BoxPSD.isChecked():
            if 'freqs' not in locals() or 'psds' not in locals():
                freqs, psds = compute_fft_welch(self.eeg_channel_data_filt, self.fs)
            self.band_power = compute_power_bands(freqs, psds)
            self.PSD.clear()
            x_positions = np.arange(len(self.bands)) * 2
            bar_item = pg.BarGraphItem(x=x_positions, height=self.band_power, width=1.5,
                                       brush=pg.mkBrush(0, 191, 255))
            self.PSD.addItem(bar_item)
            # Ajouter les étiquettes des bandes sur l'axe des abscisses
            self.PSD.getAxis('bottom').setTicks([list(zip(x_positions, self.bands))])
            # Enregistrer dans le CSV de puissance par bande si enregistrement actif
            if self.record_data:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data_row = [timestamp] + [str(val) for val in self.band_power]
                header = ['Time', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                file_exists = os.path.isfile(self.filename)
                is_empty = file_exists and os.path.getsize(self.filename) == 0
                with open(self.filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists or is_empty:
                        writer.writerow(header)
                    writer.writerow(data_row)
        else:
            self.PSD.clear()
        # Ajuster dynamiquement le rafraîchissement en fonction de la charge
        heavy_ops = self.BoxFFT.isChecked() or self.BoxPSD.isChecked()
        many_channels = len(self.eeg_channel_indices) > 4
        base_interval = int(1000 / max(1, int(self.fps.text()) if self.fps.text() else 1))
        new_interval = base_interval
        if heavy_ops and many_channels:
            new_interval = int(base_interval * 1.5)
        elif heavy_ops or many_channels:
            new_interval = int(base_interval * 1.2)
        if new_interval != self.timer.interval():
            self.timer.setInterval(new_interval)
        # Affichage 3D des signaux EEG si l'onglet 3D est actif
        if hasattr(self, 'tabs') and self.tabs.currentIndex() == 1:
            if not hasattr(self, 'gl_lines'):
                self.gl_lines = []
            # Supprimer les anciennes courbes 3D
            for line in list(getattr(self, 'gl_lines', [])):
                self.glview.removeItem(line)
            self.gl_lines.clear()
            # Couleurs des courbes (identiques à l'affichage 2D)
            colors = [
                (1, 0, 0, 1),     # rouge
                (0, 1, 0, 1),     # vert
                (0, 0, 1, 1),     # bleu
                (0, 1, 1, 1),     # cyan
                (1, 0, 1, 1),     # magenta
                (1, 1, 0, 1),     # jaune
                (1, 1, 1, 1),     # blanc
                (1, 0.65, 0, 1)   # orange
            ]
            # Tracer les signaux EEG en 3D pour chaque canal sélectionné
            for idx, ch_idx in enumerate(self.eeg_channel_indices):
                data = self.eeg_channel_data_filt[idx, :]
                # Points 3D : (temps, amplitude, décalage du canal sur l'axe Z)
                points = np.vstack([t, data, np.full(data.shape, idx * 2000.0)]).T
                line = gl.GLLinePlotItem(pos=points, color=colors[ch_idx % len(colors)],
                                          width=2, antialias=True)
                self.glview.addItem(line)
                self.gl_lines.append(line)

    def begin_recording(self):
        """Démarre l'enregistrement des données EEG dans des fichiers CSV."""
        trial_name = self.trial_name.text().strip()
        if not trial_name:
            self.label_6.setText("Veuillez entrer un nom de test (Trial Name).")
            return
        self.record_data = True
        # Préparer le dossier et les fichiers CSV
        results_dir = r"C:\Users\Konan\Desktop\EEG_GUI"
        os.makedirs(results_dir, exist_ok=True)
        # Fichier CSV pour la puissance par bande
        self.filename = os.path.join(results_dir, f"{trial_name}_power.csv")
        with open(self.filename, 'w', newline='') as file:
            pass  # créer ou réinitialiser le fichier
        # Fichier CSV pour les données brutes EEG
        self.raw_filename = os.path.join(results_dir, f"{trial_name}_raw.csv")
        self.raw_file = open(self.raw_filename, 'w', newline='')
        self.raw_writer = csv.writer(self.raw_file)
        nchan = len(self.eeg_channels) if hasattr(self, 'eeg_channels') else 8
        header = ['Time'] + [f"Ch{i+1}" for i in range(nchan)]
        self.raw_writer.writerow(header)
        self.sample_counter = 0
        self.label_6.setText("Enregistrement des données activé...")

    def end_recording(self):
        """Arrête l'enregistrement des données."""
        self.record_data = False
        if hasattr(self, 'raw_file'):
            self.raw_file.close()
        self.label_6.setText("Enregistrement terminé.")

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowTitle("EEG Analysis Interface")
        Form.resize(1280, 800)
        self.main_layout = QtWidgets.QHBoxLayout(Form)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(20)
        # Panneau gauche (contrôles)
        self.left_panel = QtWidgets.QWidget(Form)
        self.left_panel_layout = QtWidgets.QVBoxLayout(self.left_panel)
        self.left_panel_layout.setSpacing(15)
        # Groupe Connexion
        self.group_conn = QtWidgets.QGroupBox("Connexion")
        self.group_conn_layout = QtWidgets.QGridLayout(self.group_conn)
        self.label = QtWidgets.QLabel("COM Port:")
        self.com_port = QtWidgets.QLineEdit()
        self.label_2 = QtWidgets.QLabel("Win. Size:")
        self.win_size = QtWidgets.QLineEdit()
        self.label_3 = QtWidgets.QLabel("FPS:")
        self.fps = QtWidgets.QLineEdit()
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.reset_button = QtWidgets.QPushButton("Reset")
        # Icônes sur les boutons Connect/Reset
        self.connect_button.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_DriveNetIcon))
        self.reset_button.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        # Texte d'exemple dans les champs de saisie
        self.com_port.setPlaceholderText("Ex: 3")
        self.win_size.setPlaceholderText("Ex: 10")
        self.fps.setPlaceholderText("Ex: 10")
        # Placement des widgets de connexion
        self.group_conn_layout.addWidget(self.label, 0, 0)
        self.group_conn_layout.addWidget(self.com_port, 0, 1)
        self.group_conn_layout.addWidget(self.label_2, 1, 0)
        self.group_conn_layout.addWidget(self.win_size, 1, 1)
        self.group_conn_layout.addWidget(self.label_3, 2, 0)
        self.group_conn_layout.addWidget(self.fps, 2, 1)
        self.group_conn_layout.addWidget(self.connect_button, 3, 0)
        self.group_conn_layout.addWidget(self.reset_button, 3, 1)
        self.left_panel_layout.addWidget(self.group_conn)
        # Groupe Canaux EEG
        self.group_channels = QtWidgets.QGroupBox("Canaux EEG")
        self.group_channels_layout = QtWidgets.QGridLayout(self.group_channels)
        self.BoxCh1 = QtWidgets.QCheckBox("Ch1"); self.group_channels_layout.addWidget(self.BoxCh1, 0, 0)
        self.BoxCh2 = QtWidgets.QCheckBox("Ch2"); self.group_channels_layout.addWidget(self.BoxCh2, 0, 1)
        self.BoxCh3 = QtWidgets.QCheckBox("Ch3"); self.group_channels_layout.addWidget(self.BoxCh3, 1, 0)
        self.BoxCh4 = QtWidgets.QCheckBox("Ch4"); self.group_channels_layout.addWidget(self.BoxCh4, 1, 1)
        self.BoxCh5 = QtWidgets.QCheckBox("Ch5"); self.group_channels_layout.addWidget(self.BoxCh5, 2, 0)
        self.BoxCh6 = QtWidgets.QCheckBox("Ch6"); self.group_channels_layout.addWidget(self.BoxCh6, 2, 1)
        self.BoxCh7 = QtWidgets.QCheckBox("Ch7"); self.group_channels_layout.addWidget(self.BoxCh7, 3, 0)
        self.BoxCh8 = QtWidgets.QCheckBox("Ch8"); self.group_channels_layout.addWidget(self.BoxCh8, 3, 1)
        self.left_panel_layout.addWidget(self.group_channels)
        # Groupe Options d'analyse
        self.group_analysis = QtWidgets.QGroupBox("Options d'analyse")
        self.group_analysis_layout = QtWidgets.QVBoxLayout(self.group_analysis)
        self.BoxFiltering = QtWidgets.QCheckBox("Filtering")
        self.BoxFFT = QtWidgets.QCheckBox("FFT")
        self.BoxPSD = QtWidgets.QCheckBox("PSD")
        self.BoxTime = QtWidgets.QCheckBox("Time Domain")
        # Infobulles explicatives pour chaque option
        self.BoxFiltering.setToolTip("Filtrer le signal EEG (passe-bande + notch)")
        self.BoxFFT.setToolTip("Afficher le spectre de Fourier (FFT)")
        self.BoxPSD.setToolTip("Afficher la densité spectrale de puissance (PSD)")
        self.BoxTime.setToolTip("Afficher le signal temporel (Time Domain)")
        # Icônes pour les options (fichiers requis dans ./icons)
        self.BoxFiltering.setIcon(QtGui.QIcon("icons/filter_icon.png"))
        self.BoxFFT.setIcon(QtGui.QIcon("icons/fft_icon.png"))
        self.BoxPSD.setIcon(QtGui.QIcon("icons/psd_icon.png"))
        self.BoxTime.setIcon(QtGui.QIcon("icons/time_icon.png"))
        # Ajouter les options au layout
        self.group_analysis_layout.addWidget(self.BoxFiltering)
        self.group_analysis_layout.addWidget(self.BoxFFT)
        self.group_analysis_layout.addWidget(self.BoxPSD)
        self.group_analysis_layout.addWidget(self.BoxTime)
        self.left_panel_layout.addWidget(self.group_analysis)
        # Groupe Enregistrement
        self.group_record = QtWidgets.QGroupBox("Enregistrement")
        self.group_record_layout = QtWidgets.QGridLayout(self.group_record)
        self.label_7 = QtWidgets.QLabel("Trial Name:")
        self.trial_name = QtWidgets.QLineEdit()
        self.record_button = QtWidgets.QPushButton("Start")
        self.end_record_button = QtWidgets.QPushButton("End")
        # Icônes sur les boutons Start/End
        self.record_button.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.end_record_button.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        # Placeholder pour le nom de test
        self.trial_name.setPlaceholderText("Ex: Test1")
        # Placement des widgets d'enregistrement
        self.group_record_layout.addWidget(self.label_7, 0, 0)
        self.group_record_layout.addWidget(self.trial_name, 0, 1)
        self.group_record_layout.addWidget(self.record_button, 1, 0)
        self.group_record_layout.addWidget(self.end_record_button, 1, 1)
        self.left_panel_layout.addWidget(self.group_record)
        # Espace flexible pour pousser les éléments vers le haut
        self.left_panel_layout.addStretch()
        # Ajouter le panneau gauche au layout principal
        self.main_layout.addWidget(self.left_panel)
        # Panneau droit (graphiques + statut)
        self.right_panel = QtWidgets.QWidget(Form)
        self.right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel)
        # Titre principal de l'interface
        self.label_5 = QtWidgets.QLabel("EEG Analysis")
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.right_panel_layout.addWidget(self.label_5)
        # Widget de tracé principal (pyqtgraph 2D)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Sous-graphiques 2D : signal temporel, spectre, puissance par bande
        self.TimeGraph = self.win.addPlot(row=0, col=0, colspan=2, title="Time Domain")
        self.FFTGraph = self.win.addPlot(row=1, col=0, title="PSD")
        self.PSD = self.win.addPlot(row=1, col=1, title="Power per Band")
        # Création des onglets pour basculer entre vue 2D et 3D
        self.tabs = QtWidgets.QTabWidget()
        self.tab_2d = QtWidgets.QWidget()
        self.tab_3d = QtWidgets.QWidget()
        self.tab2d_layout = QtWidgets.QVBoxLayout(self.tab_2d)
        self.tab2d_layout.addWidget(self.win)
        self.tab3d_layout = QtWidgets.QVBoxLayout(self.tab_3d)
        self.glview = gl.GLViewWidget()
        self.tab3d_layout.addWidget(self.glview)
        self.tabs.addTab(self.tab_2d, "Vue 2D")
        self.tabs.addTab(self.tab_3d, "Vue 3D")
        self.right_panel_layout.addWidget(self.tabs)
        # Label de statut en bas du panneau droit
        self.label_6 = QtWidgets.QLabel("Ready.")
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.right_panel_layout.addWidget(self.label_6)
        # Ajouter le panneau droit (avec stretch) au layout principal
        self.main_layout.addWidget(self.right_panel, stretch=1)
        # Connecter les signaux des boutons aux méthodes correspondantes
        self.connect_button.clicked.connect(Form.connect_board)
        self.reset_button.clicked.connect(Form.reset_app)
        self.record_button.clicked.connect(Form.begin_recording)
        self.end_record_button.clicked.connect(Form.end_recording)
        # Timer pour la mise à jour périodique des données
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(Form.update_data_and_graphs)
        # Transférer tous les attributs de l'interface vers l'objet Form (MainApp)
        for name, value in self.__dict__.items():
            setattr(Form, name, value)
        # Effet de fondu en apparition sur le titre
        self.title_effect = QtWidgets.QGraphicsOpacityEffect()
        self.label_5.setGraphicsEffect(self.title_effect)
        self.title_effect.setOpacity(0)
        self.title_animation = QtCore.QPropertyAnimation(self.title_effect, b"opacity")
        self.title_animation.setDuration(2000)
        self.title_animation.setStartValue(0)
        self.title_animation.setEndValue(1)
        self.title_animation.start()
        # Effet de halo lumineux pulsant sur le titre
        self.glow_effect = QtWidgets.QGraphicsDropShadowEffect()
        self.glow_effect.setColor(QtGui.QColor(0, 191, 255))
        self.glow_effect.setBlurRadius(5)
        self.glow_effect.setOffset(0)
        self.label_5.setGraphicsEffect(self.glow_effect)
        # Animation de la pulsation du halo
        self.glow_animation_up = QtCore.QPropertyAnimation(self.glow_effect, b"blurRadius")
        self.glow_animation_up.setDuration(1000)
        self.glow_animation_up.setStartValue(5)
        self.glow_animation_up.setEndValue(20)
        self.glow_animation_up.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self.glow_animation_down = QtCore.QPropertyAnimation(self.glow_effect, b"blurRadius")
        self.glow_animation_down.setDuration(1000)
        self.glow_animation_down.setStartValue(20)
        self.glow_animation_down.setEndValue(5)
        self.glow_animation_down.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self.glow_sequence = QtCore.QSequentialAnimationGroup()
        self.glow_sequence.addAnimation(self.glow_animation_up)
        self.glow_sequence.addAnimation(self.glow_animation_down)
        self.glow_sequence.setLoopCount(-1)
        self.title_animation.finished.connect(self.glow_sequence.start)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Thème futuriste (fond noir, éléments bleu ciel)
    pg.setConfigOptions(background='k', foreground='w', antialias=True, useOpenGL=True)
    app.setStyleSheet("""
        QWidget {
            color: #b1b1b1;
            background-color: #1e1e1e;
        }
        QGroupBox {
            border: 2px solid #00BFFF;
            border-radius: 10px;
            margin-top: 10px;
        }
        QGroupBox::title {
            color: #00BFFF;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        QLabel, QCheckBox, QPushButton {
            color: #ffffff;
        }
        QLineEdit {
            background-color: #505050;
            color: #ffffff;
            border: 1px solid #8f8f91;
            border-radius: 5px;
        }
        QLineEdit:focus {
            border: 1px solid #00BFFF;
        }
        QPushButton {
            background-color: #1e1e1e;
            border: 2px solid #00BFFF;
            border-radius: 10px;
            font: bold 14px;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #00BFFF;
        }
        QPushButton:pressed {
            background-color: #007F9F;
        }
        QPushButton:disabled {
            background-color: #555555;
            color: #aaaaaa;
            border: 2px solid #555555;
        }
        QTabWidget::pane {
            border: 2px solid #00BFFF;
            border-radius: 10px;
        }
        QTabBar::tab {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #00BFFF;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            padding: 5px;
            margin: 2px;
        }
        QTabBar::tab:selected {
            background-color: #00BFFF;
            color: #000000;
            margin-bottom: 0px;
        }
        QTabBar::tab:!selected {
            border-bottom: 1px solid #00BFFF;
        }
        QTabBar::tab:hover:!selected {
            background-color: #007F9F;
        }
    """)
    main_window = MainApp()
    # Icône de fenêtre personnalisée (icône d'ordinateur par défaut ici)
    main_window.setWindowIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
    main_window.show()
    sys.exit(app.exec_())
