import math
import struct
import wave

import numpy as np
from faster_whisper.audio import decode_audio


def test_decode_audio_accepts_generated_wav(tmp_path):
    sample_rate = 16000
    path = tmp_path / 'sine.wav'
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b''.join(
            struct.pack('<h', int(16000 * math.sin(2 * math.pi * 440 * sample / sample_rate)))
            for sample in range(sample_rate)
        ))

    decoded = decode_audio(str(path), sampling_rate=sample_rate)

    assert isinstance(decoded, np.ndarray)
    assert decoded.size == sample_rate
    assert np.any(decoded != 0)
