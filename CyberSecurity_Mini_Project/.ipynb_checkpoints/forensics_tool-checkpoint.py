# forensics_tool.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fft
import librosa
import librosa.display
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import csv
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Audio Loading & Visualization
# -------------------------------
def load_audio(file_path):
    """Load WAV file and return sample rate & audio data."""
    try:
        sample_rate, audio = wavfile.read(file_path)
        if len(audio.shape) > 1:  # Stereo to mono
            audio = np.mean(audio, axis=1)
        return sample_rate, audio.astype(np.float32) / np.max(np.abs(audio))  # Normalize
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def plot_waveform(audio, sr, title, save_path):
    """Plot and save waveform."""
    plt.figure(figsize=(12, 4))
    time = np.linspace(0, len(audio)/sr, len(audio))
    plt.plot(time, audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram(audio, sr, title, save_path):
    """Plot and save spectrogram."""
    plt.figure(figsize=(12, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------------------
# 2. Steganography Detection (LSB Extraction)
# -------------------------------
def extract_lsb_bits(audio, num_bits=1):
    """Extract LSBs from audio samples to detect hidden data."""
    # Convert to int16 for LSB extraction
    audio_int = (audio * 32767).astype(np.int16)
    # Extract LSBs (last 'num_bits' bits)
    bits = np.bitwise_and(audio_int, (1 << num_bits) - 1).flatten()
    return bits


def detect_steganography(bits, threshold=0.5):
    """Simple detection: Check if bits look like random data (high entropy)."""
    # Count occurrences of each bit value
    counts = np.bincount(bits)
    # Normalize to probabilities
    probabilities = counts / len(bits)
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy > threshold, entropy

def bits_to_text(bits, byte_size=8):
    """Attempt to decode bits as ASCII text."""
    try:
        # Reshape and pack bits into bytes
        trimmed = bits[:len(bits)//byte_size * byte_size]
        reshaped = trimmed.reshape(-1, byte_size)
        bytes_data = np.packbits(reshaped, axis=1).flatten()
        text = ''.join(chr(b) for b in bytes_data if 32 <= b <= 126)
        return text[:100] + '...' if len(text) > 100 else text
    except:
        return "Could not decode text"

# -------------------------------
# 3. Tampering Detection (Phase Discontinuity)
# -------------------------------
def detect_phase_discontinuity(audio, window_size=1024):
    """Detect cuts/edits by phase jumps in STFT."""
    stft = librosa.stft(audio)
    phase = np.angle(stft)
    
    # Compute phase derivative (discontinuities)
    phase_diff = np.diff(phase, axis=1)
    discontinuities = np.sum(np.abs(phase_diff) > np.pi/2, axis=0)  # Sharp jumps
    
    # Find peaks (potential edit points)
    times = librosa.frames_to_time(np.arange(len(discontinuities)), sr=22050)  # Assume 22kHz
    peaks = signal.find_peaks(discontinuities, height=np.mean(discontinuities)*2)[0]
    
    return times[peaks], discontinuities

# -------------------------------
# 4. Statistical Analysis
# -------------------------------
def compute_stats(audio, sr):
    """Compute forensic stats: ZCR, energy, entropy."""
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    avg_zcr = np.mean(zcr)
    
    energy = np.mean(librosa.feature.rms(y=audio)[0])
    
    # Spectral entropy
    stft = np.abs(librosa.stft(audio))
    entropy = -np.mean([np.sum(p * np.log2(p + 1e-10)) for p in stft / np.sum(stft, axis=1, keepdims=True)])
    
    return {
        'avg_zcr': avg_zcr,
        'avg_energy': energy,
        'spectral_entropy': entropy
    }

# -------------------------------
# 5. Main Forensic Analysis
# -------------------------------
def analyze_audio(file_path):
    """Full analysis pipeline."""
    sr, audio = load_audio(file_path)
    if audio is None:
        return None
    
    os.makedirs('reports', exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Visualizations
    plot_waveform(audio, sr, f'Waveform: {base_name}', f'reports/{base_name}_waveform.png')
    plot_spectrogram(audio, sr, f'Spectrogram: {base_name}', f'reports/{base_name}_spectrogram.png')
    
    # Steganography
    bits = extract_lsb_bits(audio)
    has_stego, stego_entropy = detect_steganography(bits)
    hidden_text = bits_to_text(bits)
    
    # Tampering
    tamper_times, disc_plot = detect_phase_discontinuity(audio)
    has_tamper = len(tamper_times) > 0
    
    # Stats
    stats = compute_stats(audio, sr)
    
    # Report data
    report = {
        'file': file_path,
        'duration': len(audio)/sr,
        'sample_rate': sr,
        'steganography_detected': has_stego,
        'stego_entropy': stego_entropy,
        'hidden_text_sample': hidden_text,
        'tampering_detected': has_tamper,
        'tamper_times': tamper_times.tolist() if has_tamper else [],
        **stats
    }
    
    # Save stats to CSV
    with open('reports/stats.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=report.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(report)
    
    # Generate PDF Report
    generate_pdf_report(report, base_name)
    
    # Plot discontinuities if tampering
    if has_tamper:
        plt.figure(figsize=(10, 4))
        plt.plot(disc_plot)
        plt.title(f'Phase Discontinuities: {base_name}')
        plt.xlabel('Time Frames')
        plt.ylabel('Discontinuity Count')
        plt.tight_layout()
        plt.savefig(f'reports/{base_name}_discontinuities.png')
        plt.close()
    
    return report

def generate_pdf_report(report, base_name):
    """Generate simple PDF report."""
    pdf_path = f'reports/{base_name}_report.pdf'
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    y = height - 50
    c.drawString(50, y, f"Forensic Report: {report['file']}")
    y -= 30
    c.drawString(50, y, f"Duration: {report['duration']:.2f}s | SR: {report['sample_rate']}Hz")
    y -= 30
    
    c.drawString(50, y, f"Steganography: {'YES' if report['steganography_detected'] else 'NO'} (Entropy: {report['stego_entropy']:.3f})")
    y -= 20
    c.drawString(50, y, f"Hidden Text: {report['hidden_text_sample']}")
    y -= 30
    
    c.drawString(50, y, f"Tampering: {'YES' if report['tampering_detected'] else 'NO'}")
    if report['tamper_times']:
        y -= 20
        c.drawString(50, y, f"Suspect Times: {', '.join([f'{t:.2f}s' for t in report['tamper_times']])}")
    y -= 30
    
    c.drawString(50, y, f"Stats - ZCR: {report['avg_zcr']:.4f} | Energy: {report['avg_energy']:.4f} | Entropy: {report['spectral_entropy']:.3f}")
    
    c.save()
    print(f"PDF report saved: {pdf_path}")

# -------------------------------
# 6. Batch Processing
# -------------------------------
def main():
    # Sample audio files (add your own WAV files to audio_files/)
    audio_files = [
        'audio_files/clean.wav',
        'audio_files/suspicious.wav'  # Assume you have test files
    ]
    
    # For demo: Create a simple test audio if files don't exist
    if not os.path.exists('audio_files'):
        os.makedirs('audio_files')
        # Generate a simple sine wave as "clean" audio
        sr = 22050
        t = np.linspace(0, 5, 5*sr)
        clean_audio = np.sin(2 * np.pi * 440 * t)  # A4 note
        wavfile.write('audio_files/clean.wav', sr, (clean_audio * 32767).astype(np.int16))
        
        # "Suspicious": Add a discontinuity
        suspicious = np.concatenate([clean_audio[:2*sr], -clean_audio[2*sr:4*sr], clean_audio[4*sr:]])
        wavfile.write('audio_files/suspicious.wav', sr, (suspicious * 32767).astype(np.int16))
    
    results = []
    for file_path in audio_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing: {file_path}")
            result = analyze_audio(file_path)
            if result:
                results.append(result)
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nBatch complete! Check 'reports/' folder for outputs.")
    print(f"CSV summary: reports/stats.csv")

if __name__ == "__main__":
    main()