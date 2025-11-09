from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
from imageio.v2 import imread
from skimage import color, transform
import joblib

# TensorFlow/Keras
try:
    import tensorflow as tf
except Exception as e:
    print(f"[DL][ERROR] TensorFlow no disponible: {e}")
    raise

# ---- Custom layers (for loading GCN models) ----
import numpy as np  # ensure np available for layer config
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class GCNLayer(layers.Layer):
    def __init__(self, units, A_norm_np=None, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.A_norm_np = np.array(A_norm_np, dtype=np.float32) if A_norm_np is not None else None

    def build(self, input_shape):
        F_in = int(input_shape[-1])
        self.w = self.add_weight(
            shape=(F_in, self.units), initializer='glorot_uniform', trainable=True, name='kernel')
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='bias')
        if self.A_norm_np is not None:
            self.A_norm = tf.constant(self.A_norm_np, dtype=tf.float32)
        else:
            # fallback identidad si no viene en el config
            self.A_norm = tf.eye(int(input_shape[1]), dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        # A_norm (N,N), inputs (B,N,F_in), w(F_in, units) -> out (B,N,units)
        support = tf.einsum('ij,bjf->bif', self.A_norm, inputs)
        out = tf.tensordot(support, self.w, axes=[[2], [0]])
        if self.use_bias:
            out = out + self.b
        if self.activation is not None:
            act = tf.keras.activations.get(self.activation)
            out = act(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'A_norm_np': None if self.A_norm_np is None else self.A_norm_np.tolist(),
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        A = config.pop('A_norm_np', None)
        if A is not None:
            A = np.array(A, dtype=np.float32)
        return cls(A_norm_np=A, **config)

# MediaPipe para landmarks (opcional, requerido para MLP/GCN)
try:
    import mediapipe as mp
    MP_HANDS = mp.solutions.hands
    MP_DRAW = mp.solutions.drawing_utils
    from mediapipe.framework.formats import landmark_pb2 as MP_LM
except Exception as e:
    MP_HANDS = None
    MP_DRAW = None
    MP_LM = None
    _MP_ERR = e


def load_labels(sidecar_base: Path, model_name: str) -> list[str] | None:
    # sidecar files next to model: labels.txt, classes.txt, classes.npy
    for fname in ("labels.txt", "classes.txt"):
        p = sidecar_base.with_name(fname)
        if p.exists():
            try:
                return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                pass
    p = sidecar_base.with_name("classes.npy")
    if p.exists():
        try:
            return np.load(str(p), allow_pickle=True).tolist()
        except Exception:
            pass
    # Heurística por nombre
    up = model_name.upper()
    if "MNIST" in up:
        # 24 letras (sin J ni Z)
        return [
            "A","B","C","D","E","F","G","H","I",
            "K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"
        ]
    # ASL por defecto -> A..Z + tokens especiales
    return [chr(ord('A')+i) for i in range(26)] + ["DEL","NOTHING","SPACE"]


def preprocess_image(image_path: str, input_shape: tuple[int, int, int], model_name: str) -> np.ndarray:
    img = imread(image_path)
    if img.ndim == 2:
        gray = img.astype('float32')
        gray = (gray - gray.min()) / max(1e-8, (gray.max() - gray.min()))
    else:
        # rgb2gray fuente consistente
        gray = color.rgb2gray(img)
    H, W, C = input_shape
    # Heurística: si es ResNet (nombre contiene RESNET o tamaño grande/3 canales), usar preprocess_input
    use_resnet = ("RESNET" in model_name.upper()) or (C == 3 and max(H, W) >= 200)

    if use_resnet:
        target = (224, 224)
        # Preparar RGB 3 canales
        if img.ndim == 2:
            # expand to RGB
            rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            rgb = img.astype('float32')
            if rgb.max() > 1.5:
                rgb /= 255.0
        rgb = transform.resize(rgb, target, mode='reflect', anti_aliasing=True)
        # Keras preprocess for resnet50 expects float and RGB in range [-, +]
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
        x = resnet_pre((rgb*255.0).astype('float32'))
        x = x.astype('float32')
    else:
        # Redimensionar a tamaño esperado del modelo
        target = (H, W)
        if C == 1:
            g = transform.resize(gray, target, mode='reflect', anti_aliasing=True).astype('float32')
            x = g[..., None]
            # escalar 0..1
            x = (x - x.min()) / max(1e-8, (x.max() - x.min()))
        else:
            # 3 canales a partir de gray si fuese necesario
            if img.ndim == 2:
                rgb = np.stack([gray, gray, gray], axis=-1)
            else:
                rgb = img.astype('float32')
                if rgb.max() > 1.5:
                    rgb /= 255.0
            rgb = transform.resize(rgb, target, mode='reflect', anti_aliasing=True)
            x = rgb.astype('float32')
    return x


def is_landmark_model(model_name: str) -> bool:
    up = model_name.upper()
    return up.startswith("MLP_") or up.startswith("GCN_") or up in ("MLP_ASL.KERAS", "GCN_ASL.KERAS")


def extract_landmarks(image_path: str):
    if MP_HANDS is None:
        print(f"[DL][ERROR] MediaPipe no disponible: {_MP_ERR}")
        return None, None
    img_bgr = None
    try:
        # leer con imageio y convertir a BGR para dibujado posterior
        img = imread(image_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        img = img.astype('uint8')
        if img.max() <= 1:
            img = (img * 255).astype('uint8')
        img_rgb = img
        img_bgr = img_rgb[:, :, ::-1].copy()
    except Exception as e:
        print(f"[DL][ERROR] No se pudo leer imagen para landmarks: {e}")
        return None, None

    with MP_HANDS.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1) as hands:
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            print("[DL][WARN] No se detectaron manos")
            return img_bgr, []
        # ordenar por mayor área (spread) para priorizar mano dominante
        hands_with_area = []
        for lm in results.multi_hand_landmarks[:2]:
            pts = np.array([[p.x, p.y, getattr(p, 'z', 0.0)] for p in lm.landmark], dtype=np.float32)
            rng = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
            area = float(rng[0] * rng[1])
            hands_with_area.append((area, pts))
        hands_with_area.sort(key=lambda t: t[0], reverse=True)
        hands_lm = [t[1] for t in hands_with_area]
        return img_bgr, hands_lm


def _normalize_hand(pts: np.ndarray) -> np.ndarray:
    # pts: (21,3) en coords normalizadas [0,1]
    if pts is None or pts.size == 0:
        return np.zeros((21, 3), dtype=np.float32)
    ref = pts[0].copy()  # wrist como origen
    centered = pts - ref
    # escalar por el máximo rango 2D para invarianza de escala
    min_xy = centered[:, :2].min(axis=0)
    max_xy = centered[:, :2].max(axis=0)
    scale = float(np.linalg.norm(max_xy - min_xy))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    centered[:, :2] /= scale
    centered[:, 2] = centered[:, 2]  # mantener Z relativa
    return centered


def build_features_for_mlp(hands_lm: list[np.ndarray], expected_dim: int) -> np.ndarray:
    # Normalizar manos detectadas (hasta 2)
    hands_n = [
        _normalize_hand(h) for h in (hands_lm + [None, None])[:2]
    ]
    # Posibles vectores: 1 mano (x,y), 1 mano (x,y,z), 2 manos (x,y), 2 manos (x,y,z)
    cand = []
    one_xy = hands_n[0][:, :2].reshape(-1)
    one_xyz = hands_n[0].reshape(-1)
    two_xy = np.concatenate([hands_n[0][:, :2].reshape(-1), hands_n[1][:, :2].reshape(-1)])
    two_xyz = np.concatenate([hands_n[0].reshape(-1), hands_n[1].reshape(-1)])
    cand.extend([one_xy, one_xyz, two_xy, two_xyz])
    for v in cand:
        if v.shape[0] == expected_dim:
            return v.astype('float32')
    # Si no coincide, ajustar por pad/trunc
    v = two_xyz  # el más completo
    if expected_dim < v.shape[0]:
        v = v[:expected_dim]
    elif expected_dim > v.shape[0]:
        v = np.pad(v, (0, expected_dim - v.shape[0]))
    return v.astype('float32')


def build_features_for_gcn(hands_lm: list[np.ndarray], nodes: int, feats: int) -> np.ndarray:
    # construir tensor (N,F) con N en {21,42}, F en {2,3}
    hands_n = [
        _normalize_hand(h) for h in (hands_lm + [None, None])[:2]
    ]
    # Elegir manos/nodos deseados
    if nodes <= 21:
        base = hands_n[0]
    else:
        base = np.concatenate([hands_n[0], hands_n[1]], axis=0)  # (42,3)
        if base.shape[0] < nodes:
            base = np.pad(base, ((0, nodes - base.shape[0]), (0, 0)))
    # Seleccionar features
    if feats <= 2:
        base = base[:, :2]
        if feats == 1:
            base = base[:, :1]
    # Ajustar shape exacta (nodes, feats)
    if base.shape[0] > nodes:
        base = base[:nodes, :]
    if base.shape[1] > feats:
        base = base[:, :feats]
    if base.shape[0] < nodes:
        base = np.pad(base, ((0, nodes - base.shape[0]), (0, 0)))
    if base.shape[1] < feats:
        base = np.pad(base, ((0, 0), (0, feats - base.shape[1])))
    return base.astype('float32')


def select_best_prediction(model, candidates: list[np.ndarray], is_gcn: bool) -> tuple[int, float]:
    best_idx, best_conf = -1, -1.0
    for x in candidates:
        try:
            y = model.predict(x, verbose=0)
            y = y[0] if isinstance(y, list) else y
            y = np.array(y)
            if y.ndim > 1:
                y = y.reshape((1, -1))
            conf = float(np.max(y[0]))
            idx = int(np.argmax(y[0]))
            if conf > best_conf:
                best_conf, best_idx = conf, idx
        except Exception as _:
            continue
    return best_idx, best_conf


def load_sidecar_scaler(model_path: Path):
    """Busca un scaler .joblib junto al modelo (mismo directorio)."""
    try:
        base = model_path.parent
        if not base.exists():
            return None
        # prioridad por nombres comunes y específicos por familia
        preferred = [
            "scaler_GCN.joblib", "scaler_MLP.joblib",
            "scaler_final.joblib", "scaler.joblib",
        ]
        for name in preferred:
            p = base / name
            if p.exists():
                print(f"[DL] Usando scaler sidecar: {p.name}")
                return joblib.load(str(p))
        # fallback: primer .joblib en carpeta
        for f in base.glob("*.joblib"):
            print(f"[DL] Usando scaler sidecar: {f.name}")
            return joblib.load(str(f))
    except Exception as e:
        print(f"[DL][WARN] No se pudo cargar scaler sidecar: {e}")
    return None


def apply_vector_scaler(vec: np.ndarray, scaler) -> np.ndarray:
    """Aplica StandardScaler si dimensiones coinciden."""
    if scaler is None:
        return vec
    try:
        # Detectar n_features esperadas
        n_exp = None
        if hasattr(scaler, 'n_features_in_'):
            n_exp = int(scaler.n_features_in_)
        elif hasattr(scaler, 'mean_'):
            n_exp = int(len(scaler.mean_))
        if n_exp is None or n_exp != vec.shape[-1]:
            return vec
        out = scaler.transform(vec.reshape(1, -1))[0]
        return out.astype('float32')
    except Exception:
        return vec


def draw_and_save_overlay(img_bgr: np.ndarray, hands_lm: list[np.ndarray]) -> str | None:
    if img_bgr is None:
        return None
    try:
        # Crear copia en RGB para dibujar con MP_DRAW
        img_rgb = img_bgr[:, :, ::-1].copy()
        if MP_DRAW is not None and MP_HANDS is not None and MP_LM is not None and len(hands_lm) > 0:
            # Construir objetos LandmarkList para dibujado
            for pts in hands_lm:
                lm_list = MP_LM.NormalizedLandmarkList(
                    landmark=[MP_LM.NormalizedLandmark(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
                )
                MP_DRAW.draw_landmarks(
                    img_rgb,
                    lm_list,
                    MP_HANDS.HAND_CONNECTIONS,
                    MP_DRAW.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    MP_DRAW.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        import tempfile
        out_path = Path(tempfile.gettempdir()) / ("overlay_" + next(tempfile._get_candidate_names()) + ".png")
        from imageio.v2 import imwrite
        imwrite(str(out_path), img_rgb)
        return str(out_path)
    except Exception as e:
        print(f"[DL][WARN] No se pudo generar overlay: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model-file', required=True)
    args = ap.parse_args()

    model_path = Path(args.model_file)
    if not model_path.exists():
        print(f"[DL][ERROR] Modelo no encontrado: {model_path}")
        return
    image_path = args.image

    # Cargar modelo Keras (estructura + pesos)
    try:
        model = tf.keras.models.load_model(str(model_path), custom_objects={'GCNLayer': GCNLayer})
    except Exception as e:
        print(f"[DL][ERROR] No se pudo cargar el modelo: {e}")
        return

    name_up = model_path.name.upper()
    if is_landmark_model(model_path.name):
        # Requiere landmarks MediaPipe
        img_bgr, hands_lm = extract_landmarks(image_path)
        if hands_lm is None:
            return
        if len(hands_lm) == 0:
            # Sin manos -> devolver NOTHING si es ASL
            labels = load_labels(model_path, model_path.name)
            if labels and "NOTHING" in [str(x).upper() for x in labels]:
                print("Resultado: NOTHING")
            else:
                print("Resultado: NOTHING")
            overlay = draw_and_save_overlay(img_bgr, hands_lm)
            if overlay:
                print(f"Overlay: {overlay}")
            return
        overlay = draw_and_save_overlay(img_bgr, hands_lm) if hands_lm else None
        # Obtener shape de entrada para decidir vector/tensor
        ish = model.input_shape
        if isinstance(ish, (list, tuple)) and len(ish) > 0 and isinstance(ish[0], (list, tuple)):
            ish = ish[0]
        # MLP: (None, D) ; GCN: (None, N, F)
        if len(ish) == 2:
            # MLP: generar variantes y elegir la de mayor confianza
            D = int(ish[1])
            scaler = load_sidecar_scaler(model_path)
            # variantes de features: centradas, crudas, y espejadas en X
            # centradas (actual)
            feats_center = build_features_for_mlp(hands_lm, D).reshape(1, -1).astype('float32')
            # crudas en [0,1] (sin centrar)
            def _flat_raw(hs):
                one_xy = hs[0][:, :2].reshape(-1)
                one_xyz = hs[0].reshape(-1)
                two_xy = np.concatenate([hs[0][:, :2].reshape(-1), hs[1][:, :2].reshape(-1)])
                two_xyz = np.concatenate([hs[0].reshape(-1), hs[1].reshape(-1)])
                return [one_xy, one_xyz, two_xy, two_xyz]
            hs_raw = [(hands_lm + [np.zeros((21,3),dtype=np.float32)])[0], (hands_lm + [np.zeros((21,3),dtype=np.float32)])[1] if len(hands_lm)>1 else np.zeros((21,3),dtype=np.float32)]
            hs_raw = [hs_raw[0], hs_raw[1]]
            cand_raw = _flat_raw(hs_raw)
            cand_raw = [v[:D] if v.shape[0]>D else np.pad(v,(0,D-v.shape[0])) for v in cand_raw]
            cand_raw = [v.reshape(1,-1).astype('float32') for v in cand_raw]
            # espejado X: x' = 1-x (en crudo)
            hs_mirror = [hs_raw[0].copy(), hs_raw[1].copy()]
            hs_mirror[0][:,0] = 1.0 - hs_mirror[0][:,0]
            hs_mirror[1][:,0] = 1.0 - hs_mirror[1][:,0]
            cand_mirror = _flat_raw(hs_mirror)
            cand_mirror = [v[:D] if v.shape[0]>D else np.pad(v,(0,D-v.shape[0])) for v in cand_mirror]
            cand_mirror = [v.reshape(1,-1).astype('float32') for v in cand_mirror]
            # aplicar scaler si disponible y dimensión casa
            candidates = [feats_center] + cand_raw + cand_mirror
            if scaler is not None:
                scaled = []
                for x in candidates:
                    v = x.reshape(1, -1)[0]
                    v2 = apply_vector_scaler(v, scaler)
                    if v2.shape[0] == D:
                        scaled.append(v2.reshape(1, -1))
                candidates = candidates + scaled
            idx, conf = select_best_prediction(model, candidates, is_gcn=False)
        elif len(ish) == 3:
            # GCN: variantes tensor crudo/centrado y espejado X
            N = int(ish[1]); F = int(ish[2])
            scaler = load_sidecar_scaler(model_path)
            # centrado
            tensor_center = build_features_for_gcn(hands_lm, N, F)
            x_center = np.expand_dims(tensor_center, axis=0).astype('float32')
            # crudo (sin centrar)
            def _build_raw(hs, N, F):
                base = (hs + [np.zeros((21,3),dtype=np.float32)])[0]
                if N>21:
                    base = np.concatenate([base, (hs + [np.zeros((21,3),dtype=np.float32)])[1]], axis=0)
                if F<=2:
                    base = base[:, :2]
                # pad/trunc a (N,F)
                if base.shape[0] < N:
                    base = np.pad(base, ((0,N-base.shape[0]),(0,0)))
                if base.shape[0] > N:
                    base = base[:N,:]
                if base.shape[1] < F:
                    base = np.pad(base, ((0,0),(0,F-base.shape[1])))
                if base.shape[1] > F:
                    base = base[:,:F]
                return base.astype('float32')
            tensor_raw = _build_raw(hands_lm, N, F)
            x_raw = np.expand_dims(tensor_raw, axis=0).astype('float32')
            # espejado X para crudo
            tensor_mirror = tensor_raw.copy()
            if F >= 1:
                tensor_mirror[:,0] = 1.0 - tensor_mirror[:,0]
            x_mirror = np.expand_dims(tensor_mirror, axis=0).astype('float32')
            candidates = [x_center, x_raw, x_mirror]
            # Añadir versiones escaladas con StandardScaler de entrenamiento si está presente (aplicado sobre vector plano)
            if scaler is not None:
                scaled = []
                for x in candidates:
                    v = x.reshape(1, -1)[0]  # (N*F)
                    v2 = apply_vector_scaler(v, scaler)
                    if v2.shape[0] == N*F:
                        scaled.append(v2.reshape(1, N, F))
                candidates = candidates + scaled
            idx, conf = select_best_prediction(model, candidates, is_gcn=True)
        else:
            print(f"[DL][ERROR] Forma de entrada no soportada para landmarks: {ish}")
            return

        if idx < 0:
            print("[DL][ERROR] No se pudo obtener predicción válida para landmarks")
            return

        labels = load_labels(model_path, model_path.name)
        if labels and 0 <= idx < len(labels):
            label = str(labels[idx])
        else:
            if idx < 26:
                label = chr(ord('A') + idx)
            else:
                label = str(idx)
        lab_up = label.upper()
        if lab_up in ("DEL", "NOTHING", "SPACE"):
            print(f"Resultado: {lab_up}")
        elif len(label) == 1 and label.isalpha():
            print(f"Resultado: {lab_up}")
        else:
            for ch in label:
                if ch.isalpha():
                    print(f"Resultado: {ch.upper()}")
                    break
            else:
                print(f"Resultado: {label}")
        if overlay:
            print(f"Overlay: {overlay}")
        return
    else:
        # Pipeline de imagen (CNN/ResNet)
        try:
            ish = model.input_shape
            if isinstance(ish, (list, tuple)) and isinstance(ish[0], (list, tuple)):
                ish = ish[0]
            H, W, C = int(ish[1]), int(ish[2]), int(ish[3])
        except Exception:
            H, W, C = 96, 96, 1
        x = preprocess_image(image_path, (H, W, C), model_path.name)
        x = np.expand_dims(x, axis=0).astype('float32')
        try:
            y = model.predict(x, verbose=0)
            if isinstance(y, list):
                y = y[0]
            y = np.array(y)
            if y.ndim > 2:
                y = y.reshape((y.shape[0], -1))
            idx = int(np.argmax(y[0]))
        except Exception as e:
            print(f"[DL][ERROR] Falló predict(): {e}")
            return
        labels = load_labels(model_path, model_path.name)
        if labels and 0 <= idx < len(labels):
            label = str(labels[idx])
        else:
            if idx < 26:
                label = chr(ord('A') + idx)
            else:
                label = str(idx)
        lab_up = label.upper()
        if lab_up in ("DEL", "NOTHING", "SPACE"):
            print(f"Resultado: {lab_up}")
        elif len(label) == 1 and label.isalpha():
            print(f"Resultado: {lab_up}")
        else:
            for ch in label:
                if ch.isalpha():
                    print(f"Resultado: {ch.upper()}")
                    break
            else:
                print(f"Resultado: {label}")


if __name__ == '__main__':
    main()
