import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
import os
import pandas as pd
from scipy.stats import skew, kurtosis
import threading
import queue
import time

# Audio playback requires simpleaudio. If it's not available, playback will be disabled.
try:
    import simpleaudio as sa
except ImportError:
    sa = None

class AudioAnalysisDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Audio Analysis Dashboard")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.audio_path = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self.current_audio = None
        self.sr = None
        self.features = {}
        self.category = tk.StringVar(value="music")
        
        # Playback object
        self.play_obj = None
        
        # Queue for thread communication
        self.queue = queue.Queue()
        
        # Create main frames & widgets
        self.create_frames()
        self.create_controls()
        self.create_visualization_area()
        self.create_feature_display()
        self.create_progress_bar()
        
        # Start checking queue for updates
        self.check_queue()
    
    def create_frames(self):
        # Top frame for controls
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill="x", padx=10, pady=10)
        
        # Middle frame for visualizations
        self.viz_frame = ttk.Frame(self.root)
        self.viz_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Bottom frame for features and progress
        self.bottom_frame = ttk.Frame(self.root, padding="10")
        self.bottom_frame.pack(fill="both", padx=10, pady=10)
        
        # Separate frame for feature display (left) and export (right)
        self.feature_frame = ttk.Frame(self.bottom_frame)
        self.feature_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.export_frame = ttk.Frame(self.bottom_frame)
        self.export_frame.pack(side="right", fill="y")
        
        # Status bar
        self.status_bar = ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", side="bottom", padx=10, pady=5)
    
    def create_controls(self):
        # File selection
        ttk.Label(self.control_frame, text="Audio File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(self.control_frame, textvariable=self.audio_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.control_frame, text="Analyze", command=self.analyze_audio).grid(row=0, column=3, padx=5, pady=5)
        
        # Category selection
        ttk.Label(self.control_frame, text="Category:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        ttk.Combobox(self.control_frame, textvariable=self.category, 
                     values=["music", "human_voice", "animal_sound"]).grid(row=0, column=5, padx=5, pady=5)
        
        # Playback controls (if simpleaudio is available)
        if sa is not None:
            ttk.Button(self.control_frame, text="Play Audio", command=self.play_audio).grid(row=0, column=6, padx=5, pady=5)
            ttk.Button(self.control_frame, text="Stop Audio", command=self.stop_audio).grid(row=0, column=7, padx=5, pady=5)
        else:
            ttk.Label(self.control_frame, text="(Audio Playback disabled: simpleaudio not installed)").grid(row=0, column=6, columnspan=2, padx=5, pady=5)
        
        # Export button
        ttk.Button(self.control_frame, text="Export Features", command=self.export_features).grid(row=0, column=8, padx=5, pady=5)
    
    def create_visualization_area(self):
        # Notebook for different visualizations
        self.notebook = ttk.Notebook(self.viz_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Define tabs: added "Chroma Features" tab for chromagram visualization.
        self.tabs = {
            "waveform": ttk.Frame(self.notebook),
            "mel_spectrogram": ttk.Frame(self.notebook),
            "mfccs": ttk.Frame(self.notebook),
            "delta_mfccs": ttk.Frame(self.notebook),
            "frequency_distribution": ttk.Frame(self.notebook),
            "chroma_features": ttk.Frame(self.notebook)
        }
        
        for name, tab in self.tabs.items():
            tab_title = name.replace("_", " ").title()
            self.notebook.add(tab, text=tab_title)
            
        # Create a figure and canvas for each tab
        self.figures = {}
        self.canvases = {}
        
        for name, tab in self.tabs.items():
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            self.figures[name] = fig
            
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            self.canvases[name] = canvas
    
    def create_feature_display(self):
        # Features treeview
        columns = ("Feature", "Value")
        self.tree = ttk.Treeview(self.feature_frame, columns=columns, show="headings", height=12)
        
        # Set column headings
        self.tree.heading("Feature", text="Feature")
        self.tree.heading("Value", text="Value")
        
        # Set column widths
        self.tree.column("Feature", width=250)
        self.tree.column("Value", width=150)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.feature_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_progress_bar(self):
        # Progress bar to indicate analysis in progress
        self.progress = ttk.Progressbar(self.export_frame, orient="horizontal", mode="indeterminate", length=200)
        self.progress.pack(pady=10)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg"), ("All Files", "*.*")]
        )
        if file_path:
            self.audio_path.set(file_path)
    
    def analyze_audio(self):
        audio_path = self.audio_path.get()
        if not audio_path or not os.path.exists(audio_path):
            self.status.set("Error: Please select a valid audio file")
            return
        
        # Clear previous data
        self.clear_visualizations()
        self.clear_features()
        
        # Start progress indication
        self.progress.start(10)
        self.status.set(f"Analyzing {os.path.basename(audio_path)}...")
        
        # Start analysis in a separate thread
        threading.Thread(target=self._analyze_thread, args=(audio_path,), daemon=True).start()
    
    def _analyze_thread(self, audio_path):
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None)
            self.current_audio = y
            self.sr = sr
            
            # Extract features (new features included)
            features = self._extract_features(y, sr, os.path.basename(audio_path))
            
            # Create visualizations
            self._create_waveform(y, sr)
            self._create_mel_spectrogram(y, sr)
            self._create_mfccs(y, sr)
            self._create_delta_mfccs(y, sr)
            self._create_frequency_distribution(y, sr)
            self._create_chroma_features(y, sr)
            
            # Update the UI
            self.queue.put(("features", features))
            self.queue.put(("status", f"Analysis complete: {os.path.basename(audio_path)}"))
        except Exception as e:
            self.queue.put(("status", f"Error: {str(e)}"))
        finally:
            self.queue.put(("progress_stop", None))
    
    def _extract_features(self, y, sr, filename):
        features = {}
        features['filename'] = filename
        features['category'] = self.category.get()
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        
        # Basic spectral features
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCCs and delta
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        
        # Additional spectral features:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Extract simple statistics
        features['mel_mean'] = float(np.mean(mel_spec_db))
        features['mel_std'] = float(np.std(mel_spec_db))
        features['mel_skew'] = float(skew(mel_spec_db.reshape(-1)))
        features['mel_kurtosis'] = float(kurtosis(mel_spec_db.reshape(-1)))
        
        # Dominant mel band analysis
        band_energies = np.sum(mel_spec, axis=1)
        features['dominant_mel_band'] = int(np.argmax(band_energies))
        
        # Energy ratios in frequency bands
        mel_bands = mel_spec.shape[0]
        low_band = int(mel_bands * 0.33)
        mid_band = int(mel_bands * 0.66)
        
        low_energy = np.sum(mel_spec[:low_band, :])
        mid_energy = np.sum(mel_spec[low_band:mid_band, :])
        high_energy = np.sum(mel_spec[mid_band:, :])
        total_energy = low_energy + mid_energy + high_energy
        
        features['low_freq_ratio'] = float(low_energy / total_energy if total_energy > 0 else 0)
        features['mid_freq_ratio'] = float(mid_energy / total_energy if total_energy > 0 else 0)
        features['high_freq_ratio'] = float(high_energy / total_energy if total_energy > 0 else 0)
        
        # MFCC statistics (first 5 coefficients)
        for i in range(min(5, mfccs.shape[0])):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i, :]))
            features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i, :]))
        
        # Temporal features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Additional spectral feature statistics
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        features['tonnetz_mean'] = float(np.mean(tonnetz))
        
        self.features = features
        return features
    
    def _create_waveform(self, y, sr):
        fig = self.figures["waveform"]
        fig.clear()
        ax = fig.add_subplot(111)
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title('Audio Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        fig.tight_layout()
        self.queue.put(("canvas_update", "waveform"))
    
    def _create_mel_spectrogram(self, y, sr):
        fig = self.figures["mel_spectrogram"]
        fig.clear()
        ax = fig.add_subplot(111)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel Spectrogram')
        fig.tight_layout()
        self.queue.put(("canvas_update", "mel_spectrogram"))
    
    def _create_mfccs(self, y, sr):
        fig = self.figures["mfccs"]
        fig.clear()
        ax = fig.add_subplot(111)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title('MFCCs')
        fig.tight_layout()
        self.queue.put(("canvas_update", "mfccs"))
    
    def _create_delta_mfccs(self, y, sr):
        fig = self.figures["delta_mfccs"]
        fig.clear()
        ax = fig.add_subplot(111)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        img = librosa.display.specshow(mfcc_delta, x_axis='time', sr=sr, ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title('Delta MFCCs')
        fig.tight_layout()
        self.queue.put(("canvas_update", "delta_mfccs"))
    
    def _create_frequency_distribution(self, y, sr):
        fig = self.figures["frequency_distribution"]
        fig.clear()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        band_energies = np.sum(mel_spec, axis=1)
        ax1.bar(range(len(band_energies)), band_energies)
        dom_band = np.argmax(band_energies)
        ax1.axvline(dom_band, color='r', linestyle='--', label=f'Dominant: {dom_band}')
        ax1.set_title('Mel Band Energy Distribution')
        ax1.set_xlabel('Mel Frequency Bands')
        ax1.set_ylabel('Energy')
        ax1.legend()
        
        mel_bands = mel_spec.shape[0]
        low_band = int(mel_bands * 0.33)
        mid_band = int(mel_bands * 0.66)
        low_energy = np.sum(mel_spec[:low_band, :])
        mid_energy = np.sum(mel_spec[low_band:mid_band, :])
        high_energy = np.sum(mel_spec[mid_band:, :])
        total_energy = low_energy + mid_energy + high_energy
        low_ratio = low_energy / total_energy if total_energy > 0 else 0
        mid_ratio = mid_energy / total_energy if total_energy > 0 else 0
        high_ratio = high_energy / total_energy if total_energy > 0 else 0
        
        ax2.pie([low_ratio, mid_ratio, high_ratio], 
                labels=['Low Freq', 'Mid Freq', 'High Freq'], 
                autopct='%1.1f%%',
                colors=['#66b3ff', '#99ff99', '#ff9999'])
        ax2.set_title('Frequency Energy Distribution')
        fig.tight_layout()
        self.queue.put(("canvas_update", "frequency_distribution"))
    
    def _create_chroma_features(self, y, sr):
        fig = self.figures["chroma_features"]
        fig.clear()
        ax = fig.add_subplot(111)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr, ax=ax, cmap='coolwarm')
        fig.colorbar(img, ax=ax)
        ax.set_title('Chroma Features')
        fig.tight_layout()
        self.queue.put(("canvas_update", "chroma_features"))
    
    def check_queue(self):
        try:
            while True:
                message = self.queue.get_nowait()
                action, data = message
                
                if action == "status":
                    self.status.set(data)
                elif action == "features":
                    self.update_feature_display(data)
                elif action == "canvas_update":
                    self.canvases[data].draw()
                elif action == "progress_stop":
                    self.progress.stop()
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)
    
    def update_feature_display(self, features):
        self.clear_features()
        # Define important features to display
        key_features = [
            ('filename', 'Filename'),
            ('category', 'Category'),
            ('duration', 'Duration (s)'),
            ('dominant_mel_band', 'Dominant Mel Band'),
            ('low_freq_ratio', 'Low Freq Ratio'),
            ('mid_freq_ratio', 'Mid Freq Ratio'),
            ('high_freq_ratio', 'High Freq Ratio'),
            ('mel_mean', 'Mel Spec Mean (dB)'),
            ('zcr_mean', 'Zero Crossing Rate'),
            ('rms_mean', 'RMS Energy'),
            ('tempo', 'Tempo (BPM)'),
            ('spectral_centroid_mean', 'Spectral Centroid'),
            ('spectral_bandwidth_mean', 'Spectral Bandwidth'),
            ('spectral_rolloff_mean', 'Spectral Rolloff'),
            ('spectral_contrast_mean', 'Spectral Contrast'),
            ('tonnetz_mean', 'Tonnetz')
        ]
        for i in range(1, 6):
            key_features.append((f'mfcc_{i}_mean', f'MFCC {i} Mean'))
        
        for key, label in key_features:
            if key in features:
                value = features[key]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                self.tree.insert("", "end", values=(label, value))
    
    def clear_features(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
    
    def clear_visualizations(self):
        for name, fig in self.figures.items():
            fig.clear()
            self.canvases[name].draw()
    
    def export_features(self):
        if not self.features:
            self.status.set("Error: No features to export")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{os.path.splitext(self.features.get('filename', 'audio_features'))[0]}_features.csv"
        )
        if file_path:
            try:
                pd.DataFrame([self.features]).to_csv(file_path, index=False)
                self.status.set(f"Features exported to {file_path}")
            except Exception as e:
                self.status.set(f"Error exporting features: {str(e)}")
    
    def play_audio(self):
        if self.current_audio is None or self.sr is None:
            self.status.set("Error: No audio loaded to play.")
            return
        if sa is None:
            self.status.set("Audio playback is disabled (simpleaudio not installed).")
            return
        try:
            # Normalize and convert to 16-bit PCM for playback
            audio_norm = self.current_audio / np.max(np.abs(self.current_audio))
            audio_pcm = (audio_norm * 32767).astype(np.int16)
            self.play_obj = sa.play_buffer(audio_pcm, 1, 2, self.sr)
            self.status.set("Playing audio...")
        except Exception as e:
            self.status.set(f"Error playing audio: {str(e)}")
    
    def stop_audio(self):
        if self.play_obj:
            self.play_obj.stop()
            self.status.set("Audio playback stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalysisDashboard(root)
    root.mainloop()
