import torch
import torchaudio
import os
from os.path import join
import multiprocessing as mp
from scipy.signal import medfilt

def lowpass_median_create(args):
    file_name, idx, total, input_folder, output_folder = args
    fullpath = join(input_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)
    
    waveform, sample_rate = torchaudio.load(fullpath)   # 讀取原始音檔

    print(f"{idx + 1}/{total} file：{fullpath} | Shape: {waveform.size()} | Sample Rate: {sample_rate}")
    
    waveform_new = torchaudio.functional.lowpass_biquad(waveform,sample_rate,1000)  # 低通濾波 (此時為音訊為 2D tensor，waveform.shape = (channel, sample_length))
    waveform_new = waveform_new.numpy() # 轉為 numpy: <class 'numpy.ndarray'>
    waveform_new = medfilt(waveform_new[0], 1001)   # 取音訊的第一聲道的音訊數列，變為 1D ，才可進行均值濾波 (此音檔本來就只有單聲道)
    waveform_new = torch.tensor(waveform_new, dtype=torch.float32).unsqueeze(0)  # 變回 2D tensor, e.g.[1, 22050]，明確標記為「1 個聲道」的資料，若不加.unsqueeze(0)，會變成 1D tensor, e.g. [22050] 只有一串聲音資料，但沒有標記聲道數，無法存成音檔
    
    # Notes:
    # torch.tensor 為新版寫法，舊版為 torch.FloatTensor: 
    # waveform_new = torch.FloatTensor(waveform_new).unsqueeze(0)
    
    torchaudio.save(join(output_folder, file_name), waveform_new, sample_rate)  # 輸出新音檔


if __name__ == '__main__':
    input_folders = ['abnormal', 'normal']
    output_folders = [join('output', 'lowpass_abnormal_1khz_median'), join('output', 'lowpass_normal_1khz_median')]

    for input_folder, output_folder in zip(input_folders, output_folders):
        files = os.listdir(input_folder)
        total = len(files)
        args_list = [(f, i, total, input_folder, output_folder) for i, f in enumerate(files)]

        print(f"\nProcessing {input_folder} -> {output_folder}")

        pool = mp.Pool()
        pool.map(lowpass_median_create, args_list)
        pool.close()
        pool.join()

    print("\nAll files processed.")
