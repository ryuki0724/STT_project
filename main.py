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
        vad_mode=2):
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
                    input_device_index=3,  # 指定設備ID
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
    if device.type in ['cpu', 'mps']:
        model = whisper.load_model(model_path)
        start_time = time.time()
        result = model.transcribe(load_audio(audio_bytes))
        end_time = time.time()
        print(f'解碼耗時: {end_time - start_time} 秒')
        text = result["text"]
    else:
        model = whisper.load_model(model_path).to(device)
        audio_data = whisper.pad_or_trim(load_audio(audio_bytes))
        mel = whisper.log_mel_spectrogram(audio_data).to(device)

        start_time = time.time()
        # 檢測語言
        _, probs = model.detect_language(mel)
        end_time = time.time()
        print(f"語言檢測耗時: {end_time - start_time} 秒")
        print(f"檢測到的語言: {max(probs, key=probs.get)}")

        start_time = time.time()
        # 解碼音頻
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        text = result.text
        end_time = time.time()
        print(f'解碼耗時: {int(end_time - start_time)} 秒')

    print(f"識別文本: {text}")
    return speak(text, get_language_code(text))


def speak(text, lang):
    # 將文本轉換為語音
    tts = gTTS(text=text, lang=lang)

    # 將臨時文件保存到 'audios' 目錄中
    with tempfile.NamedTemporaryFile(delete=False, dir='audios/') as temp_file:
        temp_file.name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        tts.save(temp_file.name)

        # 加載 mp3 文件並播放
        audio = AudioSegment.from_file(temp_file.name, format="mp3")
        play(audio)

        # os.remove(temp_file.name)


def get_device():
    """自動檢測並返回最佳可用的設備"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'  # 支援 Apple Silicon (M1/M2) 的 Metal 加速
    else:
        return 'cpu'


def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    print("\n可用的音頻輸入設備：")
    for i in range(0, numdevices):
        device = p.get_device_info_by_host_api_device_index(0, i)
        if device.get('maxInputChannels') > 0:
            print(f"設備ID {i}: {device.get('name')}")

    p.terminate()


while True:
    # 只在程序開始時列出一次音頻設備
    if 'audio_devices_listed' not in globals():
        list_audio_devices()
        audio_devices_listed = True
    byte_data = record_until_silence()
    device = get_device()
    print(f"使用設備: {device}")
    process_audio(byte_data, device=device, model_path='models/base.pt')
