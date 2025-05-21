# 模型訓練原理

本專案模型訓練流程：

* step 0. 將每段血管聲音週期轉成 spectrogram，組成 batch 丟進模型

* step 1. 將 每一張 spectrogram 的 label 進行 one-hot 編碼

* step 2. 每一張 spectrogram 會經過 ResNet 得到 logits，代表模型對於分類血管正常與狹窄的信心分數

* step 3. 用 softmax 將 logits 轉成機率

* step 4. 計算 Binary Cross Entropy Loss

* step 5. 反向傳播

以下將用簡單例子說明 step1~4 的運作機制

---
### step 1. Label One-Hot 編碼
假如 labels 存放三筆資料對應的 label，0 為血管正常，1 為血管狹窄
```
labels = [0, 1, 1]
```
對三個 label 進行 one hot 編碼後，存放於 onehot_labels
```
onehot_labels = [[1, 0],
                 [0, 1],
                 [0, 1]]
```
---
### step 2. 模型輸出 logits
Logits 是模型最後一層的輸出，代表每個類別的原始信心分數。
```
logits = [[2.0, 0.5],   # 第一筆模型覺得第 0 類比較有信心
          [0.1, 3.0],   # 第二筆模型覺得第 1 類比較有信心
          [1.5, 1.5]]   # 第三筆模型兩類信心一樣
```
---
### step 3. logits 經過 softmax 函數轉換
需要利用 softmax 將 Logits 轉成機率，後續才能配合 one-hot label，計算 Binary Cross Entropy Loss。
```
probabilities = [[0.8176, 0.1824],  # 第一筆預測為「正常」的機率是 81.8%，
                 [0.0522, 0.9478],  # 第二筆預測為「狹窄」機率是 94.8%，
                 [0.5000, 0.5000]]  # 第三筆預測「正常」與「狹窄」的機率各為 50%。
```
---
### step 4. 計算 Loss
根據模型預測的機率，和真實標籤的差距，計算出一個 loss 數值，loss 值越小越準確。
Binary Cross Entropy Loss 公式如下：
```
Loss = - [y * log(p) + (1 - y) * log(1 - p)]
```

計算三筆資料的 Binary Cross Entropy Loss：
```
# 第一筆資料的 loss
Loss1 = - (1 * log(0.8176) + 0 * log(1 - 0.1824)) ≈ -log(0.8176) ≈ 0.201

# 第二筆資料的 loss
Loss2 = - (0 * log(0.0522) + 1 * log(0.9478)) ≈ -log(0.9478) ≈ 0.053

# 第三筆資料的 loss
Loss3 = - (0 * log(0.5) + 1 * log(0.5)) = -log(0.5) ≈ 0.693

# 三筆 loss 平均
Loss = (Loss_1 + Loss_2 + Loss_3) / 3 ≈ 0.316
```
---
### Notes
如果用 nn.CrossEntropyLoss() 的話，就不需要自己做 softmax，因為它會內建處理 logit → softmax → log loss。但因為此專案使用 BCELoss() 搭配 one-hot label，所以才要加上 softmax 以對應 label 格式。

<!-- ---
假設你有以下設定：
假設 batch_size = 3: 一個 batch 拿 3 筆 spectrogram 資料

以下以一個batch的流程做說明:

對應的標籤是：[0, 1, 1]

每筆 spectrogram 是 1×40×100 的大小（1通道、40個頻率、100個時間點）

batch_size = 3

---
### 1. Label One-Hot 編碼範例

使用 `torch.nn.functional.one_hot` 將數值標籤轉換為 one-hot 向量格式，配合 BCELoss 使用。

```python
import torch
import torch.nn.functional as F

# 假設 labels_tmp 存放三筆資料對應的 label 
images.shape = (3, 1, 40, 100)   # 3張圖片、1通道、40個頻率、100個時間點
labels_tmp = torch.tensor([0, 1, 1])

# 將 labels_tmp 中每一個 label 做 one-hot 編碼
labels = F.one_hot(labels_tmp, num_classes=2).float().to(device)

# 輸出結果
# labels = tensor([[1., 0.],
#                  [0., 1.],
#                  [0., 1.]])
```
## 2. 模型前向傳播
把 spectrogram 圖片（images）丟進模型 model，模型會跑一輪 forward，輸出每筆資料的 logits
* Logits 是模型最後一層的輸出，它們本身不是機率，只是代表每個類別的原始信心分數，我們會再用 softmax 把它轉成機率，才能配合 one-hot label 計算 loss。
```python
model = resnet50(weights=None)
outputs = model(images)

# 輸出結果
# outputs = [[2.1, 0.3], 
#            [0.5, 1.7], 
#            [1.2, 1.2]]   # shape = (batch_size, 2)

softmax = nn.Softmax(dim=1).to(device)
outputs = softmax(outputs)

# 輸出結果
# outputs = [[0.8176, 0.1824],
#            [0.0522, 0.9478],
#            [0.5000, 0.5000]]

```

代表第一筆預測「0 類」的機率是 81.8%，第二筆預測 1 類機率是 94.8%，第三筆不確定，兩邊各 50%。

## 3. 計算 loss
➡️「根據模型預測的機率，和真實標籤的差距，計算出一個 loss 數值」
➡️ 這個 loss 表示模型在這個 batch 預測得好不好，值越小越準確。
```python
criterion = nn.BCELoss().to(device)
loss_list = []

loss = criterion(outputs,labels)
loss_list.append(loss.item())   # .item() 是 PyTorch 的寫法，意思是把 loss 這個 Tensor 轉成純數字（float）

# loss ≈ 0.316
```
### Binary Cross Entropy Loss 計算：
```
Binary Cross Entropy Loss 公式
Loss = - [y * log(p) + (1 - y) * log(1 - p)]

第一筆資料的 loss
Loss1 = - (1 * log(0.8176) + 0 * log(1 - 0.1824)) ≈ -log(0.8176) ≈ 0.201

第二筆資料的 loss
Loss2 = - (0 * log(0.0522) + 1 * log(0.9478)) ≈ -log(0.9478) ≈ 0.053

第三筆資料的 loss
Loss3 = - (0 * log(0.5) + 1 * log(0.5)) = -log(0.5) ≈ 0.693

此範例 loss
Loss = (Loss_1 + Loss_2 + Loss_3) / 3 ≈ 0.316
``` -->












