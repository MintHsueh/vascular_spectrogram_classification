import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join, isdir
import random
import copy
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
import torchvision.transforms as transforms


# 回傳 normal 與 abnormal 病人編號的聯集與交集 (之後 test set 的病人會從交集中取，其餘病人則當 train set)
def get_total_patient(normal_folder_path,abnormal_folder_path):
    normal_files = os.listdir(normal_folder_path)
    normal_patient = set()
    
    for file_name in normal_files:
        if(file_name[-3:] != 'WAV'):
            continue
        name = str()
        for char in file_name:
            if(char == '_'):
                break
            name += char
        normal_patient.add(name)
    
    abnormal_files = os.listdir(abnormal_folder_path)
    abnormal_patient = set()

    for file_name in abnormal_files:
        if(file_name[-3:] != 'WAV'):
            continue
        
        name = str()
        for char in file_name:
            if(char == '_'):
                break
            name += char
        abnormal_patient.add(name)
    
    return (abnormal_patient | normal_patient),(abnormal_patient & normal_patient)

# 從原始資料夾裡讀進 patient_list 中病人編號的所有 wav 音檔，轉成 waveform 格式並回傳
def load_wavs(folder_path,patient_list):
        files = os.listdir(folder_path)
        random.shuffle(files)
        
        ret_files = list()
        file_paths = list()
        for file_name in files:
            if(file_name[-3:] != 'WAV'):
                continue
            
            file_path = join(folder_path, file_name)
        
            if isdir(file_path):
                continue
            
            name = str()
            for char in file_name:
                if(char == '_'):
                    break
                name += char

            if name not in patient_list:
                continue

            wav, _ = torchaudio.load(file_path)
            ret_files.append(wav)
            file_paths.append(file_path)

        return copy.deepcopy(ret_files),copy.deepcopy(file_paths)

# 建立訓練集與測試集
class trainset(Dataset):
    def __init__(self,normal_wavs,abnormal_wavs):
        self.normal_imgs   = self.transform_img(normal_wavs)    # 1D 時域訊號轉換為 2D 時頻譜
        self.abnormal_imgs = self.transform_img(abnormal_wavs)  # 1D 時域訊號轉換為 2D 時頻譜

        self.imgs = list()  # self.imgs = [{'img': <Tensor>, 'label': 0 或 1}, {'img': <Tensor>, 'label': 0 或 1},...]

        for img in self.normal_imgs:
            info = dict()
            info['img']   = img
            info['label'] = 0   # normal 資料 label 為 0
            self.imgs.append(info)

        for img in self.abnormal_imgs:
            info = dict()
            info['img']   = img
            info['label'] = 1   # abnormal 資料 label 為 1
            self.imgs.append(info)

        random.shuffle(self.imgs)   # 不讓模型先看到一堆正常再看到一堆異常，避免學習順序偏差

    def __getitem__(self,index):
        img   = self.imgs[index]['img']
        label = self.imgs[index]['label']
        return img,label

    def __len__(self):
        return len(self.imgs)
    
    def transform_img(self,wavs_files):
        imgs = list()
        spectrogram_func = torchaudio.transforms.Spectrogram(n_fft=440,power=1).to(device)  # n_fft=440 代表每次用窗長為 440 個樣本點來轉換, power=1 代表值為振幅

        # 對每一段聲音做標準化 → 轉成頻譜圖 → 截前 40 個頻率 → 補齊長度 → 進資料集
        for wav in wavs_files:
            max_wav = wav.max(1)
            min_wav = wav.min(1)
            wav *= 1/(max_wav.values[0]-min_wav.values[0])  # 振幅最小最大差值的正規化
            wav = wav.to(device)
            img = spectrogram_func(wav) # 將 waveform (1D時域) 用短時傅立葉轉換 STFT 轉換為 Spectrogram (2D時頻譜) 
            img = img[:,:40,:]  # 裁切成前 40 個頻率範圍 (只保留了從 0Hz ~ 1950Hz 的聲音頻率範圍)
            img = img.cpu()
            img = F.interpolate(img, size=100)  # 圖片統一大小 (時間軸都補成100)
            imgs.append(img)
        return copy.deepcopy(imgs)  # img.shape = (1, 40, 100)
    
        # Notes:
        # sample rate = 22050 → 一秒採集 22050 個點，決定「最多能聽到多高頻」
        # n_fft = 440 → 一次抓 440 個時間點，來分析這段聲音中包含哪些頻率成分 (回傳 440 個頻率 bin)，決定「要切多細的頻率刻度」
        # 一個 bin 對應一個頻率區間，範圍是 sample_rate / n_fft Hz = = 22050 / 440 = 50.11Hz，即第 0 bin 為 0 Hz，第 1 bin 為 50，第 439 個 bin 為 22050 Hz
        
def main():
    normal_file_path    =  'normal_peak_cut_1k_selected'      
    abnormal_file_path  =  'abnormal_peak_cut_1k_selected'      

    patient_list_total,patient_list_intersection = get_total_patient(normal_file_path,abnormal_file_path)
    patient_list_intersection = list(patient_list_intersection) # 同時有 normal 與 abnormal 資料的病人編號
    patient_list_total        = list(patient_list_total)    # 所有病人編號

    softmax = nn.Softmax(dim=1).to(device)  # softmax 函數，dim = 1 為在每一筆資料的類別維度上做 softmax (outputs 的 shape = (3, 2) → 表示有 3 筆資料，每筆資料有 2 類別的分數)

    num_repeats = 2 # for test
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []

    for round in range(num_repeats):
        print("\n------------------------------------------")
        print(f"Round {round + 1} training...")

        # Part 1. 建立資料集與測試集
        random.shuffle(patient_list_intersection)   # 打亂同時有正常及異常資料的病人順序，為了讓 test set 每次取不同病人資料，做 cross-validation
        
        patient_test_list  = patient_list_intersection[:3]  # test set 的病人編號 (取 3 個病人當 test set)
        test_normal_wavs   , test_normal_wavs_paths     = load_wavs(normal_file_path,patient_test_list)     # test set 病人的所有 normal 音檔的 waveform 及路徑
        test_abnormal_wavs , test_abnormal_wavs_paths   = load_wavs(abnormal_file_path,patient_test_list)   # test set 病人的所有 abnormal 音檔的 waveform 及路徑

        patient_train_list  = list(set(patient_list_total)-set(patient_list_intersection[:3]))  # train set 的病人編號 (除了 test set 那 3 個病人以外的病人 當 train set)
        train_normal_wavs   , train_normal_wavs_paths   = load_wavs(normal_file_path,patient_train_list)     # train set 病人的所有 normal 音檔的 waveform 及路徑
        train_abnormal_wavs , train_abnormal_wavs_paths = load_wavs(abnormal_file_path,patient_train_list)   # train set 病人的所有 abnormal 音檔的 waveform 及路徑

        print('\ntrain set patient:', patient_train_list, ', test set patient:', patient_test_list)

        train_dataset = trainset(train_normal_wavs[:len(train_abnormal_wavs)],train_abnormal_wavs)  # 建立 train dataset (時頻譜img + label 的清單)
        test_dataset  = trainset(test_normal_wavs[:len(test_abnormal_wavs)],test_abnormal_wavs)     # 建立 test dataset (時頻譜img + label 的清單)

        print(f"\nTrain Set")
        print("normal files count:  ", len(train_dataset.normal_imgs))
        print("abnormal files count:", len(train_dataset.abnormal_imgs))
        print(f"\nTest Set")
        print("normal files count:  ", len(test_dataset.normal_imgs))
        print("abnormal files count:", len(test_dataset.abnormal_imgs),"\n")

        # Part 2. 創建 ResNet50 模型
        model = resnet50(weights=None)  # 載入 torchvision 內建的 ResNet50 模型架構 (weights=None 表示不使用 ImageNet 預訓練權重，從頭訓練)
        model.fc = nn.Linear(2048,2)    # 原本 ResNet50 預設是 1000 類 → 改為輸出 2 類（二分類）
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)   # conv1 的 in_channels=3 改為 1，因為 spectrogram 為灰階 (原本是RGB 彩圖)
        model = model.to(device)

        num_epochs = 1  # 原設 50  # 設定訓練幾 epoch
        batch_size = 8  # 原設 80  # 一次丟給模型的小批量資料
        learning_rate = 0.001   # 控制模型參數更新的幅度
        print_interval = 20     # 每 20 step 顯示一次訓練結果（loss, acc, 預估剩餘時間）
        # save_folder = 'resnet50/'

        criterion = nn.BCELoss().to(device) # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(),lr = learning_rate)   # 使用 Adam 優化器來根據 loss 更新模型參數

        # DataLoader: 把資料集切成多個 batch
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)   # shuffle=True: 每個 epoch 前都會打亂訓練資料順序，避免模型只記憶順序特徵)
        test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)   # shuffle=False: 測試資料不打亂

        total_step = len(train_loader)  # 計算每個 epoch 要訓練幾 step
        loss_list = []  # 記錄每一筆 batch 的 loss
        acc_list = []   # 記錄每一筆 batch 的 accuracy
        start = time.time()

        # Part 3. 模型訓練
        for epoch in range(num_epochs):
            for i,(images,labels_tmp) in enumerate(train_loader):   # 每次從 train_loader 拿一個 batch 進行訓練 (batch_size 張 spectrogram 與對應的 label)
                images = images.to(device)
                
                # 把標籤轉成 one-hot 編碼
                labels = F.one_hot(labels_tmp, num_classes=2).float().to(device)

                # 模型前向傳播，得到模型輸出的機率值（兩類的機率）
                outputs = model(images)     # 模型輸出為 logits
                outputs = softmax(outputs)  # 將 logits 利用 softmax 轉為機率

                # 計算 Loss
                loss = criterion(outputs,labels)
                loss_list.append(loss.item())   # .item() 為 PyTorch 寫法，把 loss 這個 Tensor 轉成純數字（float）

                # 反向傳播
                optimizer.zero_grad()   # 每次訓練前，先把上一輪計算的梯度清空 (PyTorch 預設會累加梯度 gradient accumulation)
                loss.backward()         # 反向傳播，算出所有參數的梯度
                optimizer.step()        # 根據算出的梯度，更新模型的權重參數

                # 計算準確率
                total = labels.size(0)  # 取得這個 batch 中的樣本數 (= batch_size)
                _,predicted = torch.max(outputs.data,1) # predicted 為模型預測的類別（e.g. [0, 1, 0, 0, 1]）
                correct = 0
                for label_idx in range(len(predicted)):
                    if(labels[label_idx][predicted[label_idx]] == 1):   # labels 是 one-hot 標籤
                        correct += 1
                acc_list.append(correct/total)

                # 顯示訓練進度
                if (i+1) % print_interval == 0:
                    end = time.time()
                    remain_time = (end-start)/((epoch*total_step + i+1)/(num_epochs*total_step)) - (end-start)
                    h = int(remain_time/3600)
                    remain_time %= 3600
                    m = int((remain_time)/60)
                    remain_time %= 60
                    s = int(remain_time)

                    print('Epoch[{}/{}],Step[{},{}],Loss:{:6.4f},Accuracy:{:4.2f}%, remain: {:2d}h {:2d}m {:2d}s'
                            .format(epoch+1,num_epochs,i+1,total_step,loss.item(),(correct/total)*100,h,m,s))

        # Part 4. 模型測試
        model.eval()    #  PyTorch 模型的評估模式

        with torch.no_grad(): # 做 inference，不要記錄梯度
            # 設定準確度參數(correct / total)、混淆矩陣參數 
            correct = 0
            total = 0
            normal_to_abnormal = 0
            normal_to_normal = 0
            abnormal_to_abnormal = 0
            abnormal_to_normal = 0
            
            # 測試集測試
            for images,labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # 進行預測
                outputs = model(images) # outputs 是模型輸出的 logits
                _,predicted = torch.max(outputs.data,1) # torch.max(..., 1) 會取出最大值的 index，即預測的類別（0 或 1）
                
                # 計算樣本總數
                total += labels.size(0) 

                # 建立混淆矩陣
                for i in range(len(labels)):
                    if(predicted[i] == labels[i] and labels[i] == 0):
                        normal_to_normal += 1
                    elif(predicted[i] != labels[i] and labels[i] == 0):
                        normal_to_abnormal += 1
                    elif(predicted[i] == labels[i] and labels[i] == 1):
                        abnormal_to_abnormal += 1
                    elif(predicted[i] != labels[i] and labels[i] == 1):
                        abnormal_to_normal += 1
                
                # 累加正確數量
                correct += (predicted == labels).sum().item()

        # 計算此 batch 結果
        accuracy = (normal_to_normal + abnormal_to_abnormal) / total * 100
        sensitivity = abnormal_to_abnormal / (abnormal_to_abnormal + abnormal_to_normal) * 100  # = Recall = 異常有多少被正確分類（True Positive Rate）
        specificity = normal_to_normal / (normal_to_normal + normal_to_abnormal) * 100  # 正常有多少被正確分類（True Negative Rate）

        # 存到 list 裡最後可以求平均結果
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        print(f"\nRound {round + 1} result：Acc {accuracy:.2f}%, Sen {sensitivity:.2f}%, Spe {specificity:.2f}%")

        # Part 5. 儲存訓練結果
        # torch.save(model.state_dict(), save_folder + accuracy + sensitivity + specificity +'_conv_net_model.ckpt')

        # file_set = {
        #     'test_normal_wavs'      : test_normal_wavs_paths[:len(test_abnormal_wavs)] ,
        #     'test_abnormal_wavs'    : test_abnormal_wavs_paths  ,
        #     'train_normal_wavs'     : train_normal_wavs_paths[:len(train_abnormal_wavs)]    ,
        #     'train_abnormal_wavs'   : train_abnormal_wavs_paths 
        # }

        # file_set = pd.DataFrame.from_dict(file_set, orient='index')
        # file_set = file_set.transpose()
        # file_set.to_csv(save_folder + accuracy + sensitivity + specificity + '_file_set.csv')

    # 計算每 round 結果平均
    avg_acc = sum(accuracy_list) / num_repeats
    avg_sen = sum(sensitivity_list) / num_repeats
    avg_spe = sum(specificity_list) / num_repeats

    print("\nAverage result:")
    print(f"Accuracy   : {avg_acc:.2f}%")
    print(f"Sensitivity: {avg_sen:.2f}%")
    print(f"Specificity: {avg_spe:.2f}%")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


