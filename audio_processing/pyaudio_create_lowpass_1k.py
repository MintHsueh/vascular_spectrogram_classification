import torch
import torchaudio
import os
from os.path import join
import multiprocessing as mp

def lowpass_create(args):
    file_name, idx, total, input_folder, output_folder = args
    fullpath = join(input_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)

    waveform, sample_rate = torchaudio.load(fullpath)   # 讀原始音檔

    print(f"{idx + 1}/{total} file：{fullpath} | Shape: {waveform.size()} | Sample Rate: {sample_rate}")

    waveform_new = torchaudio.functional.lowpass_biquad(waveform, sample_rate, 1000)    # 低通濾波，濾掉 1000 HZ 以上
    torchaudio.save(join(output_folder, file_name), waveform_new, sample_rate)  # 存成新音檔
    

if __name__ == '__main__':
    input_folders = ['abnormal', 'normal']
    output_folders = [join('output', 'lowpass_abnormal_1khz'), join('output', 'lowpass_normal_1khz')]

    for input_folder, output_folder in zip(input_folders, output_folders):
        files = os.listdir(input_folder)
        total = len(files)
        args_list = [(f, i, total, input_folder, output_folder) for i, f in enumerate(files)]    # 檔名、編號、總數、input資料夾、output資料夾

        print(f"\nProcessing {input_folder} -> {output_folder}")
        pool = mp.Pool()
        pool.map(lowpass_create, args_list)
        pool.close()
        pool.join()

    print("\nAll files processed.")
