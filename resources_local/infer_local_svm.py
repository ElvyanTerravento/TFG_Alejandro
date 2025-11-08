#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia local para modelos clásicos (HOG / LBP / HOG+LBP con SVM).

Uso:
    python infer_local_svm.py --image <ruta_png> --model-dir <carpeta_con_modelo>

Requisitos Python (instalar si faltan):
    pip install numpy scikit-image scikit-learn joblib imageio

Salida:
    Imprime una línea clave: "Resultado: <LABEL>"
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Dependencias científicas
try:
    import numpy as np
    from skimage import color, transform
    from skimage.feature import hog, local_binary_pattern
    import joblib
except Exception as e:
    print(f"[LOCAL INFERENCE][ERROR] Faltan dependencias: {e}")
    print("Sugerencia: pip install numpy scikit-image scikit-learn joblib")
    sys.exit(0)


def load_summary(model_dir: Path) -> dict | None:
    """Busca summary.json en el padre del dir 'models' y lo carga si existe."""
    # Si estamos en .../<MODEL_NAME>/models, mirar el padre (<MODEL_NAME>)
    parent = model_dir.parent if model_dir.name.lower() == 'models' else model_dir
    # Algunos modelos tienen summary.json a este nivel
    cand = parent / 'summary.json'
    if cand.exists():
        try:
            with cand.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def parse_descriptor_profiles(model_dir: Path, summary: dict | None) -> dict:
    """
    Devuelve un dict con perfiles a usar, p.ej.:
      { 'type': 'hog', 'hog_profile': 'ASL' }
      { 'type': 'lbp', 'lbp_profile': 'MNIST' }
      { 'type': 'hog+lbp', 'hog_profile': 'MNIST', 'lbp_profile': 'ASL' }

    Se deduce del nombre de la carpeta padre (p.ej. HOG_MNIST+LBP_ASL_ASL_SVM_C0p1) o del summary.
    """
    # Primero, intentar por summary
    if summary and 'descriptor' in summary:
        d = str(summary['descriptor']).lower()
        if 'hog' in d and 'lbp' in d:
            # Si no tenemos perfiles explícitos, inferir por nombre
            pass
        elif 'hog' in d:
            return {'type': 'hog', 'hog_profile': infer_profile_from_name(model_dir, default='ASL')}
        elif 'lbp' in d:
            return {'type': 'lbp', 'lbp_profile': infer_profile_from_name(model_dir, default='ASL')}
    # Fallback por nombre
    parent = model_dir.parent if model_dir.name.lower() == 'models' else model_dir
    name = parent.name.upper()
    # Casos: HOG_ASL_ASL_SVM..., LBP_MNIST_ASL_SVM..., HOG_MNIST+LBP_ASL_ASL_SVM...
    if name.startswith('HOG_') and '+LBP_' in name:
        # HOG_<HPROF>+LBP_<LPROF>_...  -> extraer entre HOG_ y +LBP_ y entre +LBP_ y _SVM_
        try:
            after_hog = name.split('HOG_', 1)[1]
            hog_prof = after_hog.split('+LBP_', 1)[0]
            after_lbp = after_hog.split('+LBP_', 1)[1]
            lbp_prof = after_lbp.split('_', 1)[0]
            return {'type': 'hog+lbp', 'hog_profile': hog_prof, 'lbp_profile': lbp_prof}
        except Exception:
            return {'type': 'hog+lbp', 'hog_profile': 'ASL', 'lbp_profile': 'ASL'}
    if name.startswith('HOG_'):
        try:
            hog_prof = name.split('HOG_', 1)[1].split('_', 1)[0]
            return {'type': 'hog', 'hog_profile': hog_prof}
        except Exception:
            return {'type': 'hog', 'hog_profile': 'ASL'}
    if name.startswith('LBP_'):
        try:
            lbp_prof = name.split('LBP_', 1)[1].split('_', 1)[0]
            return {'type': 'lbp', 'lbp_profile': lbp_prof}
        except Exception:
            return {'type': 'lbp', 'lbp_profile': 'ASL'}
    # por defecto, usar HOG ASL
    return {'type': 'hog', 'hog_profile': 'ASL'}


def infer_profile_from_name(model_dir: Path, default: str = 'ASL') -> str:
    parent = model_dir.parent if model_dir.name.lower() == 'models' else model_dir
    name = parent.name.upper()
    if 'MNIST' in name:
        return 'MNIST'
    if 'ASL' in name:
        return 'ASL'
    return default


def hog_params_from_profile(profile: str | None) -> dict:
    """Devuelve HOG cfg a partir del perfil 'ASL' o 'MNIST'."""
    profile = (profile or 'ASL').upper()
    if profile == 'MNIST':
        return {
            'orientations': 9,
            'pixels_per_cell': (7, 7),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'transform_sqrt': True,
            'image_size': (28, 28),  # dataset MNIST usa 28x28, 4x4 celdas
        }
    # ASL por defecto
    return {
        'orientations': 10,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (3, 3),
        'block_norm': 'L2-Hys',
        'transform_sqrt': True,
        'image_size': (96, 96),
    }


def compute_hog(gray: np.ndarray, cfg: dict) -> np.ndarray:
    # Redimensionar
    H, W = cfg.get('image_size', (96, 96))
    img = transform.resize(gray, (H, W), mode='reflect', anti_aliasing=True)
    # HOG
    vec = hog(
        img,
        orientations=cfg.get('orientations', 9),
        pixels_per_cell=cfg.get('pixels_per_cell', (8, 8)),
        cells_per_block=cfg.get('cells_per_block', (3, 3)),
        block_norm=cfg.get('block_norm', 'L2-Hys'),
        transform_sqrt=cfg.get('transform_sqrt', True),
        visualize=False,
        feature_vector=True,
    )
    return vec.astype('float32')


def lbp_params_from_profile(profile: str | None) -> dict:
    profile = (profile or 'ASL').upper()
    if profile == 'MNIST':
        return {'P': 13, 'R': 2.8, 'grid': (8, 8), 'method': 'uniform', 'image_size': (28, 28)}
    # ASL por defecto
    return {'P': 15, 'R': 4.0, 'grid': (8, 8), 'method': 'uniform', 'image_size': (96, 96)}


def compute_lbp(gray: np.ndarray, cfg: dict) -> np.ndarray:
    Ht, Wt = cfg.get('image_size', (96, 96))
    img = transform.resize(gray, (Ht, Wt), mode='reflect', anti_aliasing=True)
    # Convertir a uint8 para evitar warnings numéricos y emular entrenamiento típico
    img = (img * 255.0).clip(0, 255).astype('uint8')
    P = int(cfg.get('P', 15))
    R = float(cfg.get('R', 4.0))
    method = cfg.get('method', 'uniform')
    GX, GY = cfg.get('grid', (8, 8))
    lbp = local_binary_pattern(img, P=P, R=R, method=method)
    # Hist por cuadrícula
    n_bins = P + 2 if method == 'uniform' else int(lbp.max() + 1)
    xs = np.linspace(0, img.shape[1], GX + 1, dtype=int)
    ys = np.linspace(0, img.shape[0], GY + 1, dtype=int)
    feats = []
    for yi in range(GY):
        for xi in range(GX):
            patch = lbp[ys[yi]:ys[yi+1], xs[xi]:xs[xi+1]]
            hist, _ = np.histogram(patch.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            feats.append(hist.astype('float32'))
    return np.concatenate(feats, axis=0)


def expected_dims_hog(cfg: dict) -> int:
    H, W = cfg.get('image_size', (96, 96))
    o = int(cfg.get('orientations', 9))
    ppc = cfg.get('pixels_per_cell', (8, 8))
    cpb = cfg.get('cells_per_block', (3, 3))
    cells_x = W // ppc[0]
    cells_y = H // ppc[1]
    nbx = int(cells_x - cpb[0] + 1)
    nby = int(cells_y - cpb[1] + 1)
    if nbx < 1 or nby < 1:
        return 0
    return nbx * nby * (cpb[0] * cpb[1]) * o


def expected_dims_lbp(cfg: dict) -> int:
    P = int(cfg.get('P', 15))
    method = cfg.get('method', 'uniform')
    bins = P + 2 if method == 'uniform' else (P + 1)
    GX, GY = cfg.get('grid', (8, 8))
    return GX * GY * bins


def load_scaler(model_dir: Path):
    npz = model_dir / 'scaler_params.npz'
    if not npz.exists():
        return None
    try:
        data = np.load(str(npz), allow_pickle=True)
        # intentamos claves comunes de StandardScaler
        mean = data.get('mean') if 'mean' in data.files else data.get('mean_')
        scale = data.get('scale') if 'scale' in data.files else data.get('scale_')
        var = data.get('var') if 'var' in data.files else data.get('var_')
        return {
            'mean': mean,
            'scale': scale,
            'var': var,
        }
    except Exception as e:
        print(f"[LOCAL INFERENCE] Error cargando scaler_params.npz: {e}")
        return None


def apply_scaler(x: np.ndarray, scaler: dict | None) -> np.ndarray:
    if scaler is None:
        return x
    mean = scaler.get('mean')
    scale = scaler.get('scale')
    if mean is not None and scale is not None and len(mean) == x.shape[0]:
        x = (x - mean) / (scale + 1e-12)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model-dir', required=True)
    args = ap.parse_args()

    img_path = Path(args.image)
    model_dir = Path(args.model_dir)

    if not img_path.exists():
        print(f"[LOCAL INFERENCE][ERROR] Imagen no existe: {img_path}")
        return
    if not model_dir.is_dir():
        print(f"[LOCAL INFERENCE][ERROR] Carpeta de modelo no válida: {model_dir}")
        return

    # Cargar imagen
    from imageio.v2 import imread
    try:
        image = imread(str(img_path))
    except Exception as e:
        print(f"[LOCAL INFERENCE][ERROR] No se pudo leer la imagen: {e}")
        return
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype('float32')
        gray = (gray - gray.min()) / (max(1e-8, gray.max() - gray.min()))

    # Descubrir descriptores/perfiles a usar
    summary = load_summary(model_dir)
    desc = parse_descriptor_profiles(model_dir, summary)
    # Tamaño base por dataset de entrenamiento
    train_ds = parse_training_dataset(model_dir)
    base_size = (96, 96) if train_ds == 'ASL' else (28, 28)

    # Construir features según el tipo (con image_size del dataset)
    x_parts = []
    hog_cfg = None
    lbp_cfg = None
    if desc['type'] in ('hog', 'hog+lbp'):
        hog_cfg = hog_params_from_profile(desc.get('hog_profile'))
        hog_cfg['image_size'] = base_size
        x_parts.append(compute_hog(gray, hog_cfg))
    if desc['type'] in ('lbp', 'hog+lbp'):
        lbp_cfg = lbp_params_from_profile(desc.get('lbp_profile'))
        lbp_cfg['image_size'] = base_size
        x_parts.append(compute_lbp(gray, lbp_cfg))
    if not x_parts:
        print("[LOCAL INFERENCE][ERROR] No se pudo determinar descriptor.")
        return
    x = np.concatenate(x_parts).astype('float32')

    # Cargar scaler y aplicar
    scaler = load_scaler(model_dir)
    if scaler is not None and 'mean' in scaler and scaler['mean'] is not None:
        expected = len(scaler['mean'])
        if expected != x.shape[0]:
            print(f"[LOCAL INFERENCE][WARN] Dimensión de features {x.shape[0]} != scaler.mean {expected}. Reintentando parámetros LBP…")
            # Intento 1: si hay LBP, alternar entre perfiles ASL/MNIST manteniendo dataset size
            if desc['type'] in ('lbp', 'hog+lbp'):
                # probar con P=13 y P=15
                tried = []
                for Pval in (13, 15):
                    lbp_try = dict(lbp_cfg) if lbp_cfg else lbp_params_from_profile('ASL')
                    lbp_try['P'] = Pval
                    lbp_try['image_size'] = base_size
                    x_parts2 = []
                    if desc['type'] in ('hog', 'hog+lbp') and hog_cfg is not None:
                        x_parts2.append(compute_hog(gray, hog_cfg))
                    x_parts2.append(compute_lbp(gray, lbp_try))
                    x2 = np.concatenate(x_parts2).astype('float32')
                    tried.append((Pval, x2))
                    if len(x2) == expected:
                        x = x2
                        lbp_cfg = lbp_try
                        break
                else:
                    print("[LOCAL INFERENCE][WARN] No se pudo casar dimensión con scaler tras alternar LBP.")
        # aplicar scaler
    x = apply_scaler(x, scaler)

    # Cargar clasificador (.joblib)
    clf_file = None
    # intentamos usar el nombre de fichero del summary si existe
    if summary and summary.get('saved_clf_filename'):
        cand = model_dir / summary['saved_clf_filename']
        if cand.exists():
            clf_file = cand
    if clf_file is None:
        # buscar cualquier .joblib en la carpeta
        for f in model_dir.glob('*.joblib'):
            clf_file = f
            break
    if clf_file is None:
        print("[LOCAL INFERENCE][ERROR] No se encontró archivo .joblib en el modelo.")
        return

    try:
        clf = joblib.load(str(clf_file))
        # Predicción con estimador sklearn
        x2 = x.reshape(1, -1)
        try:
            y = clf.predict(x2)
            label = y[0]
        except Exception as e:
            # fallback vía decision_function / predict_proba si fuese necesario
            try:
                if hasattr(clf, 'decision_function'):
                    scores = clf.decision_function(x2)
                    idx = int(np.argmax(scores))
                    label = clf.classes_[idx]
                elif hasattr(clf, 'predict_proba'):
                    probs = clf.predict_proba(x2)
                    idx = int(np.argmax(probs))
                    label = clf.classes_[idx]
                else:
                    raise e
            except Exception as e2:
                print(f"[LOCAL INFERENCE][ERROR] Falló la predicción: {e2}")
                return
        print(f"Resultado: {label}")
        return
    except Exception as e:
        print(f"[LOCAL INFERENCE][WARN] No se pudo cargar el clasificador con joblib: {e}")
        # Intentar fallback lineal con coef.npy e intercept.npy
        label = predict_fallback_linear(model_dir, x)
        if label is None:
            return
        print(f"Resultado: {label}")
        return


def parse_training_dataset(model_dir: Path) -> str:
    parent = model_dir.parent if model_dir.name.lower() == 'models' else model_dir
    name = parent.name.upper()
    # Tomar token antes de '_SVM_'
    if '_SVM_' in name:
        prefix = name.split('_SVM_', 1)[0]
        # dataset es el último token del prefijo (ASL o MNIST)
        tok = prefix.split('_')[-1]
        if tok in ('ASL', 'MNIST'):
            return tok
    # Fallback: heurístico
    if 'MNIST' in name:
        return 'MNIST'
    return 'ASL'


def load_linear_svm_from_npy(model_dir: Path):
    coef = model_dir / 'coef.npy'
    inter = model_dir / 'intercept.npy'
    if not (coef.exists() and inter.exists()):
        print("[LOCAL INFERENCE][ERROR] No hay coef.npy o intercept.npy para fallback lineal.")
        return None, None, None
    W = np.load(str(coef), allow_pickle=True)
    b = np.load(str(inter), allow_pickle=True)
    # clases opcionales
    classes_npy = model_dir / 'classes.npy'
    classes_txt = model_dir / 'classes.txt'
    classes = None
    if classes_npy.exists():
        try:
            classes = np.load(str(classes_npy), allow_pickle=True).tolist()
        except Exception:
            classes = None
    if classes is None and classes_txt.exists():
        try:
            classes = [ln.strip() for ln in classes_txt.read_text(encoding='utf-8').splitlines() if ln.strip()]
        except Exception:
            classes = None
    if classes is None:
        # deducir por dataset
        ds = parse_training_dataset(model_dir)
        if ds == 'MNIST':
            # En este proyecto las clases MNIST se mapearon a 24 letras (sin J ni Z)
            classes = [
                "A","B","C","D","E","F","G","H","I",
                "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"
            ]
        else:
            # ASL estándar: A-Z + del, nothing, space
            classes = [chr(ord('A') + i) for i in range(26)] + ['DEL', 'NOTHING', 'SPACE']
    return W, b, classes


def predict_fallback_linear(model_dir: Path, x: np.ndarray):
    W, b, classes = load_linear_svm_from_npy(model_dir)
    if W is None:
        return None
    x2 = x.reshape(-1)
    # Acomodar shapes
    W = np.array(W)
    b = np.array(b)
    # Ajustar clases a número de filas de W si procede
    if W.ndim == 2 and classes is not None:
        n_classes = W.shape[0]
        if len(classes) != n_classes:
            # Reajustar: truncar o rellenar
            if len(classes) > n_classes:
                classes = classes[:n_classes]
            else:
                classes = classes + [str(i) for i in range(len(classes), n_classes)]
    if W.ndim == 1:
        # binario: w*x + b, umbral 0
        s = float(np.dot(W, x2) + (b if b.ndim == 0 else b[0]))
        idx = 1 if s >= 0 else 0
        if len(classes) >= 2:
            return str(classes[idx])
        return str(idx)
    else:
        # multiclase OvR: scores = W x + b (cada fila una clase)
        # Intento de reconciliar dimensión si no coincide
        if x2.shape[0] != W.shape[1]:
            # Dimension esperada
            exp = W.shape[1]
            print(f"[LOCAL INFERENCE][WARN] Reconciliando features ({x2.shape[0]}) contra W ({exp}). Intentando alternar LBP P…")
            # Heurística: probar P=13 o P=15 con dataset inferido
            parent = model_dir.parent if model_dir.name.lower() == 'models' else model_dir
            name = parent.name.upper()
            # Inferir tipo de descriptor para decidir si probar LBP
            try_lbp = ('LBP_' in name) or ('+LBP_' in name)
            if try_lbp:
                # Construir nuevamente features intentando ambas P con base_size por dataset
                ds = parse_training_dataset(model_dir)
                base_size = (96, 96) if ds == 'ASL' else (28, 28)
                # Detectar perfiles	n = parse_descriptor_profiles(model_dir, None)
                desc = parse_descriptor_profiles(model_dir, None)
                from imageio.v2 import imread as _imread  # evitar conflictos
                # No tenemos la imagen aquí; no podemos recomputar. Aviso y continúo.
                print("[LOCAL INFERENCE][WARN] No es posible recomputar features en fallback sin acceso a la imagen.")
            # Si no se pudo, abortar con matmul normal (probablemente fallará pero dejamos mensaje claro)
        scores = W @ x2
        if b.ndim == 1 and b.shape[0] == scores.shape[0]:
            scores = scores + b
        idx = int(np.argmax(scores))
        # Asegurar índice válido
        if idx < 0 or idx >= len(classes):
            print(f"[LOCAL INFERENCE][WARN] Índice de clase {idx} fuera de rango (0..{len(classes)-1}). Aplicando módulo para mapear.")
            if len(classes) > 0:
                idx = idx % len(classes)
            else:
                return None
        # Normalizar tokens especiales
        lab = str(classes[idx])
        up = lab.upper()
        if up in ('DEL', 'NOTHING', 'SPACE'):
            return up
        # Si es una sola letra o dígito devolver tal cual
        if len(lab) == 1:
            return lab.upper()
        # Fallback: primera letra mayúscula
        for ch in lab:
            if ch.isalpha():
                return ch.upper()
        return lab


if __name__ == '__main__':
    main()
