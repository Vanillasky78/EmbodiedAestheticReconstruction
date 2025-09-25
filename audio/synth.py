# audio/synth.py
import io
import numpy as np
from scipy.io.wavfile import write as wavwrite

def embedding_to_tones(emb: np.ndarray, dur_s: float = 3.0, sr: int = 44100) -> bytes:
    emb = emb if emb.size else np.zeros(66, np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    base = np.array([220.0, 261.63, 293.66, 329.63, 392.0, 440.0])
    n_voices = 4
    chunk = max(1, emb.size // n_voices)
    freqs, amps = [], []
    for i in range(n_voices):
        seg = emb[i * chunk : (i + 1) * chunk]
        weight = float(np.mean(np.abs(seg)))
        note = base[int((np.mean(seg) * 1000) % len(base))]
        freqs.append(note * (1 + 0.5 * weight))
        amps.append(0.25 + 0.5 * weight)

    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    wave = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        wave += a * np.sin(2 * np.pi * f * t)

    env_len = int(0.1 * sr)
    env = np.linspace(0, 1, env_len)
    envelope = np.ones_like(t)
    envelope[:env_len] = env
    envelope[-env_len:] = env[::-1]
    wave = (wave * envelope) / max(np.max(np.abs(wave)), 1e-6)

    buf = io.BytesIO()
    wavwrite(buf, sr, (wave * 32767).astype(np.int16))
    return buf.getvalue()
