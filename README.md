# 🤖 Machine Learning Assignments

> 機器學習課程作業集，涵蓋從頭實作神經網路、邏輯迴歸、情感分析等經典 ML 任務。

---

## 📁 專案列表

### 🚢 Titanic 生存預測

#### `titanic_deep_nn.py` — 手刻 5 層深度神經網路
- 使用 **NumPy 從頭實作**前向傳播與反向傳播，不依賴任何 ML 框架
- 5 層架構：6 → 5 → 4 → 3 → 2 → 1 節點
- 使用 ReLU activation + Sigmoid 輸出層
- 訓練準確率達 **84.3%**

`Python` `NumPy` `Deep Learning` `Backpropagation`

---

#### `titanic_level1.py` — 邏輯迴歸從頭實作
- 手動實作資料前處理、One-hot encoding、特徵正規化
- 使用梯度下降訓練邏輯迴歸模型（degree 1 & 2 多項式特徵）
- Degree 2 訓練準確率達 **82.6%**

`Python` `Logistic Regression` `Gradient Descent`

---

#### `titanic_level2.py` — sklearn 高階實作
- 使用 **pandas + sklearn** 完成同樣任務
- 實作 StandardScaler 標準化、PolynomialFeatures 多項式特徵
- Degree 3 訓練準確率達 **87.6%**

`Python` `pandas` `scikit-learn`

---

### 🏠 `boston_housing_competition.py` — 波士頓房價預測
- 使用 **XGBoost** 進行房價回歸預測
- 手動調整超參數（learning rate、max_depth、subsample 等）
- 使用 early stopping 防止 overfitting

`Python` `XGBoost` `Regression` `Hyperparameter Tuning`

---

### 🎬 `submission.py` — 電影評論情感分析
- 實作 **Word Feature Extraction** 與 **Character N-gram Features**
- 使用梯度下降訓練二元情感分類器（正面 / 負面評論）
- 在真實電影評論資料集（polarity dataset）上訓練與評估
- 訓練誤差 < 4%，驗證誤差 < 30%

`Python` `NLP` `Sentiment Analysis` `SGD`

---

### 📧 `validEmailAddress_2.py` — Email 地址分類器
- 自行設計 10 維 feature vector，識別合法 Email 地址
- 處理引號、連續點、特殊字元等邊界情況

`Python` `Feature Engineering` `Classification`

---

## 🛠️ 環境需求

```bash
pip install numpy pandas scikit-learn xgboost
```

---

## 👤 作者

**陳思翰 (Hank Chen)**
- GitHub: [@hankchen0629](https://github.com/hankchen0629)
- Email: hank880629@gmail.com
