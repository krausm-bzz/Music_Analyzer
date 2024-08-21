import os
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

CHROMA_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
        return redirect(url_for('index'))

    file = request.files['audio_file']
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, sr=None)

        # BPM-Erkennung
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)  # Umwandlung in einen Python-Skalar
        bpm_rounded = round(bpm)

        # Tonart-Erkennung
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonart_index = np.argmax(np.mean(chroma, axis=1))
        tonart = CHROMA_KEYS[tonart_index]

        # Lautstärkeanalyse: RMS und Umwandlung in dB
        rms_mean = float(np.mean(librosa.feature.rms(y=y)))
        rms_db_rounded = round(20 * np.log10(rms_mean))

        return render_template('index.html',
                               bpm=bpm_rounded,
                               tonart=tonart,
                               rms=rms_db_rounded)
    finally:
        delete_file(filepath)

if __name__ == '__main__':
    app.run(debug=True)
