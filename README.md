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


## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/MarcOBL012/CAPTCHA-Solver-Optical-Character-Recognition-with-Machine-Learning.git
cd captcha-solver-uni
```

### 2. Install dependencies
```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-learn selenium
```

### 3. WebDriver Setup
- Download the ChromeDriver that matches your Chrome version and place it in your PATH, or specify its location in the scraping script.

---

## 📖 Usage

### 1. Download CAPTCHA Images (Scraping)
```bash
python scraping.py
```

This script uses Selenium to navigate to the target URL and save CAPTCHA images to `Captchas/`.

### 2. Train and Evaluate the Model
```bash
python codigo.py
```

What the script does:
- Loads and preprocesses images.
- Segments each CAPTCHA into individual character images.
- Applies data augmentation to balance & diversify training samples.
- Trains Random Forest and MLP classifiers.
- Outputs classification reports and confusion matrices (saved as PNGs).

---

## 🧠 Methodology

### Preprocessing
- Convert images to grayscale.
- Apply adaptive thresholding to binarize.
- Remove horizontal noise lines using morphological operations or Hough Line Transform.

### Segmentation
- Detect contours with `cv2.findContours`.
- Filter bounding boxes by size/shape to discard non-character blobs.
- Crop and normalize characters to a fixed frame (e.g. 20×20 px).

### Feature Extraction
- Resize each segmented character to **20×20 pixels**.
- Flatten into a 1D vector (400 features).
- Normalize feature values to range [0, 1].

### Training & Evaluation
- Models: **Random Forest** and **MLPClassifier**.
- Hyperparameter tuning with `GridSearchCV` and cross-validation.
- Metrics: accuracy, confusion matrix, precision/recall per class.

## 📬 Contact
If you use or extend this project, please add a note in the README or contact:

Marco Obispo — marco.obispo.l@uni.pe

