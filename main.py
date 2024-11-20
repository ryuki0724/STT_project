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


def list_microphones():
    # 列出所有可用的麥克風設備
    p = pyaudio.PyAudio()
    info = {}  # 改用字典來存儲設備信息
    print("\n可用的麥克風設備：")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # 只列出輸入設備
            print(f"[{i}] {device_info['name']}")
            info[i] = device_info  # 使用實際的設備索引作為鍵
    p.terminate()
    return info


def record_until_silence(
        input_device_index=None,  # 指定輸入設備索引
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
                    input_device_index=input_device_index,  # 指定輸入設備
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

    # 返回字節數據
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
        raise RuntimeError(f"載入音頻失敗: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# 將音頻文件處理成文字
def process_audio(audio_bytes, model, device='cpu'):
    # 确保在返回结果前检查文本是否为空
    # result = model.transcribe(load_audio(audio_bytes))
    # if not result["text"].strip():  # 检查文本是否为空
    #     raise ValueError("没有检测到任何语音内容")

    if device.type == 'cpu':
        start_time = time.time()
        
        audio_data = load_audio(audio_bytes)
        
        result = model.transcribe(
            audio_data,
            fp16=False,
        )
        end_time = time.time()
        print(f'解碼耗時: {end_time - start_time} 秒')
        print(result["text"])
        
        return speak(result["text"], get_language_code(result["text"]))
    else:
        audio_data = whisper.pad_or_trim(load_audio(audio_bytes))
        mel = whisper.log_mel_spectrogram(audio_data).to(device)

        start_time = time.time()
        _, probs = model.detect_language(mel)
        end_time = time.time()
        print(f"語言檢測耗時: {end_time - start_time} 秒")
        print(f"檢測到的語言: {max(probs, key=probs.get)}")

        start_time = time.time()
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        end_time = time.time()

        print(f'解碼耗時: {end_time - start_time} 秒')
        print(result.text)

        return speak(result.text, get_language_code(result.text))


def speak(text, lang):
    # 將文字轉換為語音
    tts = gTTS(text=text, lang=lang)

    # 將語音保存到臨時文件中
    with tempfile.NamedTemporaryFile(delete=False, dir='audios/') as temp_file:
        temp_file.name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
        tts.save(temp_file.name)

        # 加載 mp3 文件並播放
        audio = AudioSegment.from_file(temp_file.name, format="mp3")
        play(audio)

        # 刪除臨時文件
        # os.remove(temp_file.name)


def main():
    # 列出所有麥克風
    mic_dict = list_microphones()

    # 讓用戶選擇麥克風
    selected_mic = None
    while selected_mic is None:
        try:
            index = int(input("\n請選擇麥克風編號 (輸入數字): "))
            # 直接檢查輸入的索引是否在字典中
            if index in mic_dict.keys():
                selected_mic = index
                break  # 成功選擇後退出循環
            else:
                print(f"無效的選擇，可用的選項為: {list(mic_dict.keys())}")
        except ValueError:
            print("請輸入有效的數字")

    print(f"已選擇麥克風: {mic_dict[selected_mic]['name']}")  # 添加確認信息

    # 初始化設備
    device = torch.device('cpu')
    print("正在加載模型...")
    model = whisper.load_model('models/base.pt').to(device)
    print("模型加載完成！")

    try:
        while True:
            byte_data = record_until_silence(input_device_index=selected_mic)
            process_audio(byte_data, model, device)
    except KeyboardInterrupt:
        print("\n程序已停止")


if __name__ == '__main__':
    main()
