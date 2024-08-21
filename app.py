import numpy as np
import librosa
import librosa.display
from flask import Flask, render_template, request, redirect, url_for
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

upload_folder = 'static/uploads'
os.makedirs(upload_folder, exist_ok=True)

CHROMA_KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['audio_file']

    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    y, sr = librosa.load(filepath, sr=None)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo[0]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    tonart_index = np.argmax(chroma_mean)
    tonart = CHROMA_KEYS[tonart_index]

    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    return render_template('index.html',
                           bpm=bpm,
                           tonart=tonart,
                           rms=rms_mean)


if __name__ == '__main__':
    app.run(debug=True)
