# 即時語音轉寫與回應系統

這是一個基於 Whisper 的即時語音轉寫系統，能夠實時將語音轉換為文字，並通過 gTTS 生成語音回應。

## 功能特點

- 即時語音識別
- 多語言支持
- 自動語言檢測
- 語音回應功能
- 支持多種麥克風設備選擇

## 系統要求

- Python 3.8 或更高版本
- FFmpeg
- 可用的麥克風設備

## 安裝步驟

1. 克隆專案：

```bash
git clone https://github.com/ryuki0724/STT_project.git
```

2. 安裝依賴：


```bash
pip install -r requirements.txt
```

3. 下載模型文件：
   - 從以下雲端連結下載模型文件：
   - https://drive.google.com/drive/folders/1bVYhV34xWZyMT4_UmHkrYKuVKLi_Vs_1?usp=drive_link
   - 將下載的模型文件放置在專案根目錄的 `models` 資料夾中

4. 創建必要的目錄：

```bash
mkdir audios models
```

## 使用方法

1. 運行主程序：

```bash
python main.py
```

2. 選擇麥克風設備：
   - 程序啟動後會列出所有可用的麥克風設備
   - 輸入對應的設備編號進行選擇

3. 開始使用：
   - 程序會自動檢測語音輸入
   - 檢測到語音時會自動開始錄音
   - 停止說話後會自動結束錄音並進行處理
   - 系統會輸出識別的文字並播放語音回應

4. 結束程序：
   - 按 `Ctrl+C` 結束程序

## 目錄結構

```
STT_project/
├── audios/ # 臨時音頻文件目錄
├── models/ # 模型文件目錄
├── main.py # 主程序
├── requirements.txt # 依賴
└── README.md # 說明文檔
```

## 注意事項

- 確保系統已正確安裝並配置 FFmpeg
- 確保麥克風設備正常工作
- 音頻文件會暫時保存在 `audios` 目錄中
- 首次運行時需要下載並加載模型，可能需要一些時間

## 授權

本專案採用 MIT 授權條款。任何人都可以自由使用、修改和分發本軟體，無論是用於商業還是非商業用途。

詳細授權條款請參見 [MIT License](https://opensource.org/licenses/MIT)。

## 貢獻

歡迎提交 Issue 和 Pull Request。
