# CAPTCHA-Solver-Optical-Character-Recognition-with-Machine-Learning

This project implements an automated system capable of **segmenting and recognizing characters in distorted text-based CAPTCHA images**.  
The complete pipeline includes **web scraping, preprocessing, segmentation, data augmentation, and supervised Machine Learning classification**.


## 🚀 Key Features

### 🔁 1. Automated Data Collection  
- Web scraping using **Selenium** to build a labeled CAPTCHA dataset.

### 🖼️ 2. Image Preprocessing (OpenCV)  
- Grayscale conversion  
- Adaptive thresholding  
- Noise removal and interference line cleaning

### ✂️ 3. Character Segmentation  
- Contour detection (`cv2.findContours`)  
- Morphological operations  
- Extraction of individual characters via bounding boxes

### 🧪 4. Data Augmentation  
- Random rotations  
- Translations  
- Noise injection  
- Improves model robustness on distorted CAPTCHA characters

### 🤖 5. Machine Learning Classification  
- Supervised training using:
  - **Random Forest Classifier**
  - **MLPClassifier (Multilayer Perceptron)**
- Evaluation with accuracy metrics and confusion matrices

### 🔧 6. Model Optimization  
- Hyperparameter tuning with **GridSearchCV**  
- Cross-validation and performance comparison

---

## 🛠️ Tech Stack

| Category | Tools |
|---------|-------|
| Language | Python |
| Data Collection | Selenium WebDriver |
| Image Processing | OpenCV (cv2), NumPy |
| Machine Learning | scikit-learn (RandomForest, MLP, GridSearchCV) |
| Visualization | Matplotlib, Seaborn |
