import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from collections import Counter
import warnings
from sklearn.exceptions import ConvergenceWarning

# ================================
# CONFIGURACIÓN
# ================================
INPUT_DIR = "Captchas"
SEGMENT_DIR = os.path.join(INPUT_DIR, "segments_exact")
BORDER_CROP = 2
MIN_SAMPLES_PER_CLASS = 15

# ================================
# SEGMENTACIÓN DE CAPTCHA
# ================================
def segmentar_captchas():
    os.makedirs(SEGMENT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.png'):
            continue
        text = os.path.splitext(fname)[0]
        img_path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bright = cv2.convertScaleAbs(gray, alpha=1.0, beta=50)

        cols_nonwhite = np.where(np.any(bright < 255, axis=0))[0]
        if cols_nonwhite.size == 0:
            continue
        x_min, x_max = cols_nonwhite.min(), cols_nonwhite.max()
        cropped = bright[:, x_min:x_max+1]

        n_chars = len(text)
        char_w = int(cropped.shape[1] / n_chars)
        dst_folder = os.path.join(SEGMENT_DIR, text)
        os.makedirs(dst_folder, exist_ok=True)

        for i in range(n_chars):
            x1 = i * char_w
            x2 = x1 + char_w
            segment = cropped[:, x1:x2]

            mask = cv2.threshold(segment, 254, 255, cv2.THRESH_BINARY_INV)[1]
            pts = cv2.findNonZero(mask)
            if pts is None:
                char_img = segment
            else:
                x, y, w2, h2 = cv2.boundingRect(pts)
                char_img = segment[y:y+h2, x:x+w2]

            out_path = os.path.join(dst_folder, f"{i+1:02d}_{text[i]}.png")
            cv2.imwrite(out_path, char_img)

# ================================
# AUMENTO Y CARGA DE DATOS
# ================================
def augmentar(img):
    h, w = img.shape
    resultado = [img]
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rot = cv2.warpAffine(img, M, (w, h), borderValue=255)
        resultado.append(rot)
    ruido = img + np.random.normal(0, 15, img.shape).astype(np.uint8)
    ruido = np.clip(ruido, 0, 255)
    resultado.append(ruido)
    return resultado

def cargar_datos():
    imgs, lbls = [], []
    for folder in sorted(os.listdir(SEGMENT_DIR)):
        fpath = os.path.join(SEGMENT_DIR, folder)
        if not os.path.isdir(fpath): continue
        for img_path in sorted(glob(os.path.join(fpath, "*.png"))):
            label = os.path.basename(img_path).split("_")[1].split(".")[0]
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            h, w = img.shape
            img = img[BORDER_CROP:h-BORDER_CROP, BORDER_CROP:w-BORDER_CROP]
            for aug in augmentar(img):
                imgs.append(aug)
                lbls.append(label)
    return imgs, lbls

def centrar_y_vectorizar(imgs):
    max_h = max(img.shape[0] for img in imgs)
    max_w = max(img.shape[1] for img in imgs)
    vectores = []
    for img in imgs:
        canvas = np.ones((max_h, max_w), dtype=np.uint8) * 255
        h, w = img.shape
        y_off = (max_h - h) // 2
        x_off = (max_w - w) // 2
        canvas[y_off:y_off+h, x_off:x_off+w] = img
        vectores.append(canvas.flatten().astype(np.float32))
    return np.array(vectores, dtype=np.float32)

def filtrar_por_clase(X, y, minimo=MIN_SAMPLES_PER_CLASS):
    conteo = Counter(y)
    indices = [i for i, lbl in enumerate(y) if conteo[lbl] >= minimo]
    return X[indices], y[indices]

# ================================
# MODELOS Y EVALUACIÓN
# ================================
def entrenar_y_evaluar(X, y):
    X, y = shuffle(X, y, random_state=42)
    X, y = filtrar_por_clase(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Random Forest
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    rf_params = {'rf__n_estimators': [200], 'rf__max_depth': [None, 20]}
    rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    # MLP
    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=1000, early_stopping=False, random_state=42))
    ])
    mlp_params = {'mlp__hidden_layer_sizes': [(128,), (64, 64)], 'mlp__activation': ['relu'], 'mlp__alpha': [0.0001]}
    mlp_grid = GridSearchCV(mlp_pipeline, mlp_params, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    mlp_grid.fit(X_train, y_train)

    # Evaluación
    def evaluar(nombre, modelo):
        y_pred = modelo.predict(X_test)
        print(f"\n=== {nombre} ===")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f"Matriz de Confusión - {nombre}")
        plt.xlabel("Predicho"); plt.ylabel("Real")
        plt.tight_layout(); plt.show()

    evaluar("Random Forest", rf_grid.best_estimator_)
    evaluar("MLP", mlp_grid.best_estimator_)

    print("\n🎯 Mejores parámetros RF:", rf_grid.best_params_)
    print("🤖 Mejores parámetros MLP:", mlp_grid.best_params_)

# ================================
# EJECUCIÓN
# ================================
warnings.filterwarnings("ignore", category=ConvergenceWarning)

segmentar_captchas()
imagenes, etiquetas = cargar_datos()
X_vect = centrar_y_vectorizar(imagenes)
y_vect = np.array(etiquetas, dtype=str)
entrenar_y_evaluar(X_vect, y_vect)
