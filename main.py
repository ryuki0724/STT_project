import ffmpeg
import numpy as np
import torch
import whisper
import time
import pyaudio
import webrtcvad
import collections
import tempfile
import langid
import io
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime


def get_language_code(text):
    # 檢測文本的語言
    lang, _ = langid.classify(text)
    return lang


def record_until_silence(
        # 執行錄音，直到檢測到靜音
        rate=48000,
        chunk_duration_ms=30,
        padding_duration_ms=2000,
        vad_mode=3):
    # 設置語音活動檢測(VAD)
    vad = webrtcvad.Vad(vad_mode)

    # 計算每個音頻塊的大小（以幀為單位）
    chunk_size = int(rate * chunk_duration_ms / 1000)
    padding_size = int(rate * padding_duration_ms / 1000)

    # 設置 pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording...")

    num_padding_chunks = padding_size // chunk_size
    silent_chunks = collections.deque(maxlen=num_padding_chunks)
    frames = []

    while True:
        chunk = stream.read(chunk_size)
        is_speech = vad.is_speech(chunk, rate)

        if not is_speech:
            silent_chunks.append(chunk)
        else:
            # 檢測到語音，延長錄音
            frames.extend(silent_chunks)
            silent_chunks.clear()
            frames.append(chunk)

        if len(silent_chunks) == num_padding_chunks:
            # 在填充持續時間內未檢測到語音，停止錄音
            break

    print("Finished recording.")

    # 停止並關閉音頻流
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 將字節數據轉換為音頻文件
    audio = AudioSegment(
        data=b''.join(frames),
        sample_width=p.get_sample_size(pyaudio.paInt16),
        frame_rate=rate,
        channels=1
    )
    # 導出為 wav 格式
    byte_stream = io.BytesIO()
    audio.export(byte_stream, format='wav')
    byte_stream.seek(0)
    byte_data = byte_stream.read()

    return byte_data


def load_audio(file: str | bytes, sr: int = 16000):
    # 將音頻文件轉換為數值數組
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None

    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# 將音頻文件處理成文字
def process_audio(audio_bytes, device='cpu', model_path='models/base.pt'):
    device = torch.device(device)
    if device.type == 'cpu':
        model = whisper.load_model(model_path)
        start_time = time.time()
        result = model.transcribe(load_audio(audio_bytes))
        end_time = time.time()
        print(f'Time taken for decode: {end_time - start_time} seconds')
        print(result["text"])
    else:
        model = whisper.load_model(model_path).to(device)
        audio_data = whisper.pad_or_trim(load_audio(audio_bytes))
        mel = whisper.log_mel_spectrogram(audio_data).to(device)

        start_time = time.time()
        # detect the spoken language
        _, probs = model.detect_language(mel)
        end_time = time.time()
        print(f"Language detection took {end_time - start_time} seconds")
        print(f"Detected language: {max(probs, key=probs.get)}")

        start_time = time.time()
        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        end_time = time.time()
        print(f'Time taken for decode: {int(end_time - start_time)} seconds')

        return speak(result.text, get_language_code(result.text))


def speak(text, lang):
    # convert text to speech
    tts = gTTS(text=text, lang=lang)

    # Save to a temporary file in the 'audios' directory
    with tempfile.NamedTemporaryFile(delete=False, dir='audios/') as temp_file:
        temp_file.name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        tts.save(temp_file.name)

        # load mp3 and play it
        audio = AudioSegment.from_file(temp_file.name, format="mp3")
        play(audio)

        os.remove(temp_file.name)


while True:
    byte_data = record_until_silence()
    process_audio(byte_data, device='cuda', model_path='models/base.pt')
