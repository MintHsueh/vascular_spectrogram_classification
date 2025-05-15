import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from scipy.signal import find_peaks
from multiprocessing import Pool

# 單個檔案處理函式：從濾波音訊中找 peak 並切出短音檔
def save_result(args):
    file_name, folder_path = args
    # print(file_name)

    # 音訊路徑：原始低通檔與中值濾波參考檔
    source_fullpath = join('output', f'lowpass_{folder_path}_1khz', file_name)
    ref_fullpath = join('output', f'lowpass_{folder_path}_1khz_median', file_name)

    # 載入音訊波形與取樣率
    src_waveform, sample_rate = torchaudio.load(source_fullpath)
    ref_waveform, sample_rate = torchaudio.load(ref_fullpath)

    # 轉為 numpy，確保使用 numpy 函式正確
    ref_np = ref_waveform.numpy()

    # 根據標準差與平均，設定門檻找 peak（對 numpy 陣列操作）
    min_power = np.std(ref_np, axis=1) * 0.9 + np.mean(ref_np, axis=1)  # min_power 為波峰的最小門檻
    peaks, _ = find_peaks(ref_np[0], height=min_power[0], distance=11000)   # peaks 為每個波峰的時間點 (sample 編號)

    # 根據 peak 的位置，剪下短音檔（相鄰 peak 間距 < 25000）
    for i in range(len(peaks) - 1):
        pre_index = peaks[i]
        nxt_index = peaks[i + 1]

        if nxt_index - pre_index < 25000:
            # 取出原始音訊對應片段並轉為 tensor 格式
            waveform_new = src_waveform[0][pre_index:nxt_index].numpy()
            waveform_new = torch.FloatTensor(waveform_new).unsqueeze(0)

            # 命名輸出檔並儲存
            cut_file_name = file_name.replace(".WAV", f"_{pre_index}-{nxt_index}.WAV")
            output_dir = join('output', folder_path + '_peak_cut_1k')
            os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(join(output_dir, cut_file_name), waveform_new, sample_rate)
    '''
    # 畫圖顯示參考波形與 peak 位置（可選）
    plt.plot(ref_waveform[0])
    for i in peaks:
        plt.axvline(x=i, color='r')
    plt.title(file_name)
    plt.tight_layout()
    # plt.show()  # optional: comment out to avoid opening too many windows
    # plt.savefig(join(output_dir, file_name.replace(".WAV", ".png")))  # optional: save plot instead
    plt.close()
    '''


if __name__ == '__main__':
    folder_paths = ['abnormal', 'normal']  
    for folder_path in folder_paths:
        input_dir = join('output', f'lowpass_{folder_path}_1khz')
        files = os.listdir(input_dir)

        args_list = [(f, folder_path) for f in files]  

        with Pool() as pool:
            pool.map(save_result, args_list)
     

