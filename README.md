# Vascular Spectrogram Classification

本專案使用 PyTorch 框架進行「聲音訊號處理」與「ResNet-50 模型訓練」，將血管聲音訊號轉換為時頻圖後進行分類，以判斷聲音屬於正常血管或異常狹窄血管。

---
## 1. 專案簡介
### 研究背景

* 洗腎患者的血管若狹窄 (堵塞)，將無法進行洗腎，因此需藉由血管造影檢查血管是否狹窄，以進行血管疏通手術。

* 由於血管造影為侵入式檢查，又因為成本問題無法普及和隨時監控，因而有此發想：利用非侵入式可攜麥克風，蒐集洗腎患者的血管聲音以做分析, 達到及時地診斷血管是否狹窄。

### 方法突破
* 有別於傳統僅使用一維時域訊號擷取統計特徵進行分類的血管音分析方法，本專案提出以血管聲音的「二維時頻圖 (spectrogram)」作為 ResNet-50 模型輸入，透過深度學習進行模型訓練與分類，以判斷血管是否狹窄。

* 本專案結合均值濾波與低通濾波，提出一套能準確定位血管聲音波峰的方法，進而將長段訊號有效切割為大量單一週期訊號，以產生充足的訓練樣本。(ResNet 模型的輸入資料為這些單一週期訊號的時頻圖，其中每一個週期定義為波峰與波峰之間的訊號段。)


## 2. 方法說明
### 蒐集聲音訊號
* 由醫師協助定位並標註需量測的血管部位，並提供該血管為狹窄 (stenosis) 或正常 (normal) 的診斷資訊。

* 本研究所用收音裝置的取樣頻率為 22050 Hz，即每秒記錄 22050 個樣本點（elements/sec）。

### 聲音訊號處理
以下步驟程式碼存放於：audio_processing/

* 低通濾波：使用 torchaudio 套件，設定截止頻率為 1000 Hz，濾除高於此頻率的成分以降低雜訊干擾。(參考 pyaudio_create_lowpass_1k.py)

* 均值濾波：使用 torchaudio 套件，透過滑動窗口對訊號進行均值處理，使整體波形更加平滑，達到大部分訊號點的振幅接近，但波峰振幅仍然保持相對高度，有助於後續的波峰定位與訊號切割。(參考 pyaudio_create_medium_filter_1k.py)

* 切割訊號：使用 scipy 套件中的 find_peaks 函數，設定振幅門檻值 (threshold) 與波峰間距 (distance)，定位經均值濾波後的波峰位置 (下圖橘色波形)，並對應至低通濾波後的原始訊號 (下圖藍色波形)，切割出每一週期的訊號片段。
註：這些週期片段會經短時傅立葉轉換（STFT）轉換為時頻圖，作為 ResNet 模型的訓練資料。(參考 pyaudio_peak_cut.py)
![切割訊號示意圖](images/cut_signal.png)

### 訓練分類模型 
以下步驟皆參考：spectrogram_classification/pyaudio_training_resnet50.py

* 建立訓練集與測試集：使用 torch 套件。資料依病人為單位進行切分，確保同一病人的多筆聲音資料不會同時出現在訓練集與測試集中。正常樣本標記為 0，狹窄樣本標記為 1。

* 產生時頻譜圖：使用 torchaudio 套件，對每個單一週期的聲音訊號進行短時傅立葉轉換（STFT），將原本一維的時域訊號轉換為二維的時頻圖，以供模型作為輸入特徵。

* 建立分類模型：使用 torchvision 套件載入內建的 ResNet-50 預訓練模型。調整模型參數的過程中，由於時頻圖為灰階圖像，需將模型的輸入通道數 (in_channels) 調整為 1。

* 模型訓練：使用 torch 套件。進行 label 的 one-hot 編碼、模型前向傳播、計算 loss (本專案使用 binary Cross Entropy Loss function)、反向傳播更新參數等步驟。

* 模型測試：以準確度 (Accuracy)、敏感度 (Sensitivity)、特異度 (Specificity)作為模型效能評估指標。


## 3. 專案結構

```
vascular_spectrogram_classification/
│
├── audio_processing/                           
│  ├── abnormal/                          # 存放洗腎患者狹窄血管的原始音檔 (WAV)
│  ├── normal/                            # 存放洗腎患者正常血管的原始音檔 (WAV)
│  ├── output/                            # 經由 python 訊號處理後的所有音檔 (WAV)
│  ├── Pyaudio_create_lowpass_1k.py       # 對正常與狹窄的音檔進行低通濾波處理
│  ├── Pyaudio_create_medium_filter_1k.py # 對正常與狹窄的音檔進行均值濾波處理
│  └── pyaudio_peak_cut.py                # 對濾波後的波型定位波峰，並切割出多個單一週期訊號
│
├── spectrogram_classification/                 
│  ├── abnormal_peak_cut_1k_selected/    # 存放狹窄血管的單一週期訊號 (WAV)
│  ├── normal_peak_cut_1k_selected/      # 存放正常血管的單一週期訊號 (WAV)
│  └── pyaudio_training_resnet50.py      # 利用 STFT 生成時頻譜，並使用 ResNet50 進行二分類訓練
│
├── README.md                            # 專案說明文件 
└── requirements.txt                     # 所需套件列表 (其中 PyTorch 相關套件為 CPU 版本)
```


## 4. 注意事項
### 關於套件版本
requirements.txt. 中的 PyTorch 相關套件為 CPU 版本
```bash
pip install -r requirements.txt
```

若使用 GPU，可選用安裝其他相容版本套件

官網選擇相容版本:  
[https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

例如 CUDA 11.8 環境：
```bash
pip install torch==2.2.2+cu118 torchaudio==2.2.2+cu118 torchvision==0.17.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 關於資料

因涉及病人隱私，僅放上少量匿名化的血管音訊樣本，作為模型測試與範例用途。

