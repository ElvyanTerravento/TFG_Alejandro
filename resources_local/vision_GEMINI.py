from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from datetime import datetime
import time
from pathlib import Path
from typing import Iterable, Optional

# =================================================
# ================= CONFIGURACIÓN =================
# =================================================

SINGLE_IMAGE_PATH = None
BATCH_DIR: Optional[str] = None

# Nombre del modelo
MODEL_NAME = "gemini-2.5-flash"

# Archivo de salida JSON para resultados batch
OUTPUT_JSON_PATH = None

HARDCODED_API_KEY: Optional[str] = None

BATCH_LIMIT: Optional[int] = 10

# Extensiones soportadas
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Reducción / limpieza de metadata
STRIP_IMAGE_METADATA = True
TARGET_SIZE = (224, 224)
FORCE_EXACT_RESIZE = True

# Usar Vertex AI (True) o API pública (False)
USE_VERTEX = True
PROJECT_ID = "gestureid"
LOCATION = "europe-west4"

try:
	from PIL import Image  
except Exception:
	Image = None


def load_api_key(env_var: str = "GEMINI_API_KEY") -> Optional[str]:
	if HARDCODED_API_KEY:
		return HARDCODED_API_KEY
	return os.getenv(env_var)


def build_default_prompt() -> str:
	return (
		"Eres un experto en visión por computador especializado en el reconocimiento del lenguaje de signos americano (ASL)."

        "Te mostraré una o varias imágenes de manos haciendo gestos en el dataset ASL Alphabet."

        "Objetivo: determinar con precisión qué letra del alfabeto ASL representa cada gesto."

        "Instrucciones (razonamiento interno obligatorio):"

        "1. Observa cuidadosamente la imagen, y trata de obviar el fondo, centrándote únicamente en la mano."

        "2. Antes de responder, razona internamente paso a paso sobre la posición de los dedos, orientación de la palma y forma global de la mano."

        "3. Compara mentalmente el gesto con las formas estándar de las letras del alfabeto ASL."

        "4. Evalúa posibles confusiones (por ejemplo, entre “M” y “N”) y elige la más probable."

        "5. Asigna un nivel de confianza según la claridad visual (alta, media o baja)."

        "No muestres tu razonamiento: sólo la conclusión final estructurada, que debe ser únicamente la letra. "

        "En este conjunto hay 2 gestos adicionales a parte de letras, DEL (palma hacia abajo, dedos juntos y extendidos hacia abajo) y "

        "SPACE (palma hacia arriba, dedos juntos y extendidos formando una u con los dedos y el pulgar). "

        "Si no hay gesto responde NOTHING."
	)

	# return (
	# 	"¿Qué letra del lenguaje de signos americano (ASL) se muestra en esta imagen? Responde "
    # 	"únicamente con la letra o el nombre del gesto. Además, encontramos 3 gestos adicionales: "
	# 	"SPACE, DEL y NOTHING (si no se muestra ningún gesto)."
	# )


def iter_images(root: Path) -> Iterable[Path]:
	for p in root.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
			yield p


def encode_image(path: Path) -> tuple[bytes, str]:
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"No existe el archivo: {path}")
	mime, _ = mimetypes.guess_type(str(path))
	if mime is None:
		ext = path.suffix.lower()
		if ext in {".jpg", ".jpeg"}:
			mime = "image/jpeg"
		elif ext == ".png":
			mime = "image/png"
		else:
			raise ValueError("No se pudo inferir el MIME type (usa .jpg/.png)")
	# Opción de limpieza + resize
	if STRIP_IMAGE_METADATA and Image is not None:
		try:
			with Image.open(path) as im:  # type: ignore[attr-defined]
				mode = "RGB" if im.mode not in {"RGB", "L"} else im.mode
				im_c = im.convert(mode)
				if im_c.width > TARGET_SIZE[0] or im_c.height > TARGET_SIZE[1]:
					if FORCE_EXACT_RESIZE:
						im_c = im_c.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
					else:
						from io import BytesIO as _BytesIO
						scale = min(TARGET_SIZE[0] / im_c.width, TARGET_SIZE[1] / im_c.height)
						new_w = int(im_c.width * scale)
						new_h = int(im_c.height * scale)
						resized = im_c.resize((new_w, new_h), Image.Resampling.LANCZOS)
						canvas = Image.new(mode, TARGET_SIZE, (0, 0, 0) if mode == "RGB" else 0)
						off = ((TARGET_SIZE[0] - new_w) // 2, (TARGET_SIZE[1] - new_h) // 2)
						canvas.paste(resized, off)
						im_c = canvas
				from io import BytesIO
				buf = BytesIO()
				fmt = "PNG" if mime == "image/png" else "JPEG"
				save_params = {"quality": 90} if fmt == "JPEG" else {}
				im_c.save(buf, format=fmt, **save_params)
				return buf.getvalue(), mime
		except Exception:  # noqa: BLE001
			pass
	with path.open("rb") as f:
		return f.read(), mime


class GeminiClient:
	"""Abstracción pequeña para usar API pública o Vertex."""

	def __init__(self, model: str):
		self.model_name = model
		self._client = None
		self._vertex = USE_VERTEX
		if self._vertex:
			# Inicializar Vertex AI
			try:
				from google.cloud import aiplatform
				import vertexai  # type: ignore
				vertexai.init(project=PROJECT_ID, location=LOCATION)
				from vertexai.generative_models import GenerativeModel  # type: ignore
				self._client = GenerativeModel(model)
			except Exception as e:  # noqa: BLE001
				raise RuntimeError(f"Error inicializando Vertex AI: {e}")
		else:
			# API key simple
			api_key = load_api_key()
			if not api_key:
				raise RuntimeError("No se encontró GEMINI_API_KEY y HARDCODED_API_KEY es None.")
			try:
				import google.generativeai as genai  # type: ignore
				genai.configure(api_key=api_key)
				self._client = genai.GenerativeModel(model)
			except Exception as e:  # noqa: BLE001
				raise RuntimeError(f"Error configurando gemini generativeai: {e}")

	def generate(self, prompt: str, image_bytes: bytes, mime: str) -> tuple[str, dict | None]:
		"""Devuelve (texto, usage_dict|None)."""
		if self._vertex:
			# Vertex usa objetos Part para imágenes
			try:
				from vertexai.generative_models import Part  # type: ignore
				part = Part.from_data(data=image_bytes, mime_type=mime)
				resp = self._client.generate_content([prompt, part])  # type: ignore[arg-type]
				text = getattr(resp, "text", None)
				if text is None:
					# fallback
					if hasattr(resp, "candidates") and resp.candidates:
						text = resp.candidates[0].content.parts[0].text  # type: ignore[index]
				usage_meta = getattr(resp, "usage_metadata", None)
				usage = None
				if usage_meta:
					usage = {
						"input_tokens": getattr(usage_meta, "prompt_token_count", None),
						"output_tokens": getattr(usage_meta, "candidates_token_count", None),
						"total_tokens": getattr(usage_meta, "total_token_count", None),
					}
				return (text.strip() if text else "", usage)
			except Exception as e:  # noqa: BLE001
				raise RuntimeError(f"Error en generación Vertex: {e}")
		else:
			try:
				# API key pública
				# Para imágenes se pasa una lista de partes
				img_part = {"mime_type": mime, "data": image_bytes}
				resp = self._client.generate_content([prompt, img_part])  # type: ignore[arg-type]
				text = getattr(resp, "text", "") or ""
				usage_meta = getattr(resp, "usage_metadata", None)
				usage = None
				if usage_meta:
					usage = {
						"input_tokens": getattr(usage_meta, "prompt_token_count", None),
						"output_tokens": getattr(usage_meta, "candidates_token_count", None),
						"total_tokens": getattr(usage_meta, "total_token_count", None),
					}
				return text.strip(), usage
			except Exception as e:  # noqa: BLE001
				raise RuntimeError(f"Error en generación Gemini API: {e}")


def classify_image(client: GeminiClient, image_path: Path, prompt: Optional[str] = None) -> dict:
	try:
		img_bytes, mime = encode_image(image_path)
	except Exception as e:  # noqa: BLE001
		return {
			"result": f"ERROR_CARGA: {image_path.name}: {e}",
			"status": "error",
			"usage": None,
			"inference_time": None,
			"model": MODEL_NAME,
		}
	user_prompt = prompt or build_default_prompt()
	try:
		t0 = time.time()
		text, usage = client.generate(user_prompt, img_bytes, mime)
		inf_time = time.time() - t0
	except Exception as e:  # noqa: BLE001
		return {
			"result": f"ERROR_API: {image_path.name}: {e}",
			"status": "error",
			"usage": None,
			"inference_time": None,
			"model": MODEL_NAME,
		}
	return {
		"result": text,
		"status": "ok",
		"usage": usage,
		"inference_time": float(inf_time),
		"model": MODEL_NAME,
	}


def run_single(client: GeminiClient) -> None:
	image_path = Path(SINGLE_IMAGE_PATH)
	if not image_path.exists():
		print(f"La ruta SINGLE_IMAGE_PATH no existe: {image_path}", file=sys.stderr)
		return
	info = classify_image(client, image_path)
	print(f"Imagen: {image_path}\nResultado: {info['result']}")
	if info.get("usage"):
		u = info["usage"]
		print(
			"Tokens -> input: {inp}, output: {out}, total: {tot}".format(
				inp=u.get("input_tokens"), out=u.get("output_tokens"), tot=u.get("total_tokens")
			)
		)
	if info.get("inference_time") is not None:
		print(f"Inference time: {info.get('inference_time'):.3f} s")


def run_batch(client: GeminiClient, directory: Path) -> None:
	if not directory.exists():
		print(f"Directorio batch no existe: {directory}", file=sys.stderr)
		return
	print(f"Procesando batch en: {directory}\nModelo: {MODEL_NAME}")
	images = list(iter_images(directory))
	if BATCH_LIMIT is not None:
		images = images[: BATCH_LIMIT]
	total = len(images)
	if total == 0:
		print("No se encontraron imágenes.")
		return
	ok = 0
	results = []
	agg_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	agg_timing = {"total_inference_seconds": 0.0, "max_inference_seconds": 0.0}
	start_time = datetime.utcnow().isoformat() + "Z"
	for idx, img in enumerate(images, 1):
		info = classify_image(client, img)
		status = info["status"]
		if status == "ok":
			ok += 1
		rel = img.relative_to(directory).as_posix()
		usage = info.get("usage") or {}
		inf_t = info.get("inference_time")
		if isinstance(inf_t, (int, float)):
			agg_timing["total_inference_seconds"] += float(inf_t)
			if float(inf_t) > agg_timing["max_inference_seconds"]:
				agg_timing["max_inference_seconds"] = float(inf_t)
		for k in agg_usage.keys():  # type: ignore[assignment]
			if usage.get(k) is not None:
				agg_usage[k] += usage.get(k, 0)
		results.append({
			"image": rel,
			"result": info["result"],
			"status": status,
			"usage": usage or None,
			"inference_time": float(inf_t) if (isinstance(inf_t, (int, float))) else None,
		})
		if idx % 10 == 0 or idx == total:
			print(f"# Progreso: {idx}/{total} ({ok} OK)", file=sys.stderr)

	avg_inf = None
	if len(images) > 0:
		avg_inf = float(agg_timing["total_inference_seconds"]) / float(len(images))

	payload = {
		"directory": str(directory),
		"model": MODEL_NAME,
		"total": total,
		"ok": ok,
		"errors": total - ok,
		"started_at": start_time,
		"finished_at": datetime.utcnow().isoformat() + "Z",
		"usage_totals": agg_usage,
		"inference_totals": {
			"total_seconds": agg_timing["total_inference_seconds"],
			"average_seconds_per_image": avg_inf,
			"max_seconds": agg_timing["max_inference_seconds"],
		},
		"results": results,
	}
	try:
		out_path = Path(OUTPUT_JSON_PATH)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with out_path.open("w", encoding="utf-8") as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)
		# Mostrar resumen incluyendo tiempos de inferencia (total, avg, max)
		total_inf = agg_timing.get("total_inference_seconds", 0.0)
		avg_inf_disp = avg_inf if (avg_inf is not None) else float("nan")
		max_inf = agg_timing.get("max_inference_seconds", 0.0)
		print(
			f"Resumen: {ok}/{total} exitosas. JSON guardado en: {out_path}. "
			f"Tokens totales -> input: {agg_usage['input_tokens']} | output: {agg_usage['output_tokens']} | total: {agg_usage['total_tokens']}. "
			f"Inferencia -> total: {total_inf:.3f}s | avg: {avg_inf_disp:.3f}s | max: {max_inf:.3f}s"
		)
	except Exception as e:  # noqa: BLE001
		print(f"Error guardando JSON: {e}", file=sys.stderr)


def main() -> int:
	try:
		client = GeminiClient(MODEL_NAME)
	except Exception as e:  # noqa: BLE001
		print(f"Error inicializando cliente Gemini: {e}", file=sys.stderr)
		return 1
	if BATCH_DIR:
		run_batch(client, Path(BATCH_DIR))
	else:
		run_single(client)
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())

