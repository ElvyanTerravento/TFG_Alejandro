from __future__ import annotations

import base64
import json
from datetime import datetime
import time
import mimetypes
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI


# ============ CONFIGURACIÓN EXPLÍCITA ============
SINGLE_IMAGE_PATH = None
BATCH_DIR: Optional[str] = None

# Modelo a usar
MODEL_NAME = "gpt-4o"

OUTPUT_JSON_PATH = None

HARDCODED_API_KEY: Optional[str] = "Solicitar al creador original"

BATCH_LIMIT: Optional[int] = 10

# Extensiones soportadas
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

STRIP_IMAGE_METADATA = True

TARGET_SIZE = (224, 224)
FORCE_EXACT_RESIZE = True

try:  # Import opcional
    from PIL import Image 
except Exception: 
    Image = None 


def load_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    if HARDCODED_API_KEY:
        return HARDCODED_API_KEY
    key = os.getenv(env_var)
    if not key:
        raise SystemExit(
            f"No se encontró la variable de entorno {env_var}. Establécela o asigna la clave a HARDCODED_API_KEY.\n"
            "PowerShell ejemplo: $env:OPENAI_API_KEY='sk-...'"
        )
    return key


def encode_image_b64(path: Path) -> tuple[str, str]:
    """Lee una imagen, elimina metadatos si está habilitado y devuelve (data_uri, mime).

    No se expone el nombre del archivo al modelo.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    if not path.is_file():
        raise ValueError(f"La ruta no es un archivo: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext in {".png"}:
            mime = "image/png"
        else:
            raise ValueError("No se pudo inferir el MIME type de la imagen (añade extensión estándar .jpg/.png)")

    raw_bytes: bytes
    if STRIP_IMAGE_METADATA and Image is not None:
        try:
            with Image.open(path) as im:
                # Convertir a RGB para eliminar perfiles/alpha si no es necesario
                mode = "RGB" if im.mode not in {"RGB", "L"} else im.mode
                im_converted = im.convert(mode)

                # Redimensionar si excede dimensiones objetivo
                if im_converted.width > TARGET_SIZE[0] or im_converted.height > TARGET_SIZE[1]:
                    if FORCE_EXACT_RESIZE:
                        im_converted = im_converted.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    else:
                        # Mantener aspecto y hacer letterbox
                        from io import BytesIO as _BytesIO  # distinto alias para evitar confusión abajo
                        scale = min(TARGET_SIZE[0] / im_converted.width, TARGET_SIZE[1] / im_converted.height)
                        new_w = int(im_converted.width * scale)
                        new_h = int(im_converted.height * scale)
                        resized = im_converted.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        # Crear lienzo negro y centrar
                        canvas = Image.new(mode, TARGET_SIZE, (0, 0, 0) if mode == "RGB" else 0)
                        offset = ((TARGET_SIZE[0] - new_w) // 2, (TARGET_SIZE[1] - new_h) // 2)
                        canvas.paste(resized, offset)
                        im_converted = canvas
                from io import BytesIO

                buf = BytesIO()
                save_params = {}
                fmt = "PNG" if mime == "image/png" else "JPEG"
                if fmt == "JPEG":
                    save_params.update({"quality": 90, "optimize": True})
                # No pasar exif, icc_profile, etc. -> se eliminan
                im_converted.save(buf, format=fmt, **save_params)
                raw_bytes = buf.getvalue()
        except Exception:  # noqa: BLE001
            with path.open("rb") as f:
                raw_bytes = f.read()
    else:
        with path.open("rb") as f:
            raw_bytes = f.read()

    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return data_uri, mime


def build_default_prompt() -> str:
    # return (
	# 	"Eres un experto en visión por computador especializado en el reconocimiento del lenguaje de signos americano (ASL)."

    #     "Te mostraré una o varias imágenes de manos haciendo gestos en el dataset ASL Alphabet."

    #     "Objetivo: determinar con precisión qué letra del alfabeto ASL representa cada gesto."

    #     "Instrucciones (razonamiento interno obligatorio):"

    #     "1. Observa cuidadosamente la imagen, y trata de obviar el fondo, centrándote únicamente en la mano."

    #     "2. Antes de responder, razona internamente paso a paso sobre la posición de los dedos, orientación de la palma y forma global de la mano."

    #     "3. Compara mentalmente el gesto con las formas estándar de las letras del alfabeto ASL."

    #     "4. Evalúa posibles confusiones (por ejemplo, entre “M” y “N”) y elige la más probable."

    #     "5. Asigna un nivel de confianza según la claridad visual (alta, media o baja)."

    #     "No muestres tu razonamiento: sólo la conclusión final estructurada, que debe ser únicamente la letra. "

    #     "En este conjunto hay 2 gestos adicionales a parte de letras, DEL (palma hacia abajo, dedos juntos y extendidos hacia abajo) y "

    #     "SPACE (palma hacia arriba, dedos juntos y extendidos formando una u con los dedos y el pulgar). "

    #     "Si no hay gesto responde NOTHING."
	# )

    return (
		"¿Qué letra del lenguaje de signos americano (ASL) se muestra en esta imagen? Responde "
    	"únicamente con la letra o el nombre del gesto. Además, encontramos 3 gestos adicionales: "
		"SPACE, DEL y NOTHING (si no se muestra ningún gesto)."
	)


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def extract_text(resp) -> str:
    # Cliente nuevo expone .output_text; si no, parse manual
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    parts: list[str] = []
    for item in getattr(resp, "output", []):
        for c in item.get("content", []):
            if c.get("type") in {"output_text", "text"}:
                txt = c.get("text")
                if txt:
                    parts.append(txt)
    return "\n".join(parts).strip()


def classify_image(client: OpenAI, image_path: Path, prompt: Optional[str] = None, raw: bool = False) -> dict:
    """Clasifica una imagen y devuelve dict con resultado y uso de tokens.

    Retorno:
      {
        'result': str,        # texto de la respuesta o mensaje de error
        'status': 'ok'|'error',
        'usage': { 'input_tokens': int, 'output_tokens': int, 'total_tokens': int } | None,
        'model': str
      }
    """
    try:
        data_uri, _ = encode_image_b64(image_path)
    except Exception as e:  # noqa: BLE001
        return {
            "result": f"ERROR_CARGA: {image_path.name}: {e}",
            "status": "error",
            "usage": None,
            "model": MODEL_NAME,
        }
    user_prompt = prompt or build_default_prompt()
    content = [
        {"type": "input_text", "text": user_prompt},
        {"type": "input_image", "image_url": data_uri},
    ]
    resp = None
    try:
        t0 = time.time()
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": content}],
        )
        inf_time = time.time() - t0
    except Exception as e:  # noqa: BLE001
        return {
            "result": f"ERROR_API: {image_path.name}: {e}",
            "status": "error",
            "usage": None,
            "inference_time": None,
            "model": MODEL_NAME,
        }
    if raw:
        text_result = str(resp)
    else:
        text_result = extract_text(resp)

    # Extraer uso de tokens de forma robusta
    usage_obj = getattr(resp, "usage", None)
    usage: Optional[dict] = None
    if usage_obj:
        if isinstance(usage_obj, dict):
            usage = {
                k: usage_obj.get(k)
                for k in ("input_tokens", "output_tokens", "total_tokens")
                if k in usage_obj
            }
        else:
            # Puede ser un objeto con atributos
            usage = {}
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                if hasattr(usage_obj, k):
                    usage[k] = getattr(usage_obj, k)
        if usage and not usage.get("total_tokens") and usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    return {
        "result": text_result,
        "status": "ok",
        "usage": usage,
        "inference_time": float(inf_time) if ('inf_time' in locals()) else None,
        "model": MODEL_NAME,
    }


def run_single(client: OpenAI) -> None:
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


def run_batch(client: OpenAI, directory: Path) -> None:
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
    # For averages
    count_with_usage = 0
    sum_input_tokens = 0
    sum_output_tokens = 0
    sum_total_tokens = 0
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
        # accumulate per-image token sums for average
        if usage:
            count_with_usage += 1
            in_tok = usage.get("input_tokens")
            out_tok = usage.get("output_tokens")
            tot_tok = usage.get("total_tokens")
            if in_tok is not None:
                sum_input_tokens += in_tok
            if out_tok is not None:
                sum_output_tokens += out_tok
            if tot_tok is not None:
                sum_total_tokens += tot_tok
        # Acumular tokens
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

    avg_tokens = None
    if count_with_usage > 0:
        avg_tokens = {
            "input_tokens_per_image": float(sum_input_tokens) / float(count_with_usage),
            "output_tokens_per_image": float(sum_output_tokens) / float(count_with_usage),
            "total_tokens_per_image": float(sum_total_tokens) / float(count_with_usage),
        }

    payload = {
        "directory": str(directory),
        "model": MODEL_NAME,
        "total": total,
        "ok": ok,
        "errors": total - ok,
        "started_at": start_time,
        "finished_at": datetime.utcnow().isoformat() + "Z",
        "usage_totals": agg_usage,
        "usage_averages": avg_tokens,
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
        # display summary including token averages and inference timing
        total_inf = agg_timing.get("total_inference_seconds", 0.0)
        avg_inf_disp = avg_inf if (avg_inf is not None) else float("nan")
        max_inf = agg_timing.get("max_inference_seconds", 0.0)
        avg_tok_disp = avg_tokens if (avg_tokens is not None) else {"input_tokens_per_image": float('nan'), "output_tokens_per_image": float('nan'), "total_tokens_per_image": float('nan')}
        print(
            f"Resumen: {ok}/{total} exitosas. JSON guardado en: {out_path}. "
            f"Tokens totales -> input: {agg_usage['input_tokens']} | output: {agg_usage['output_tokens']} | total: {agg_usage['total_tokens']}. "
            f"Tokens promedio (por imagen con usage): in={avg_tok_disp['input_tokens_per_image']:.1f}, out={avg_tok_disp['output_tokens_per_image']:.1f}, tot={avg_tok_disp['total_tokens_per_image']:.1f}. "
            f"Inferencia -> total: {total_inf:.3f}s | avg: {avg_inf_disp:.3f}s | max: {max_inf:.3f}s"
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error guardando JSON: {e}", file=sys.stderr)


def main() -> int:
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    # Batch tiene prioridad si está configurado
    if BATCH_DIR:
        run_batch(client, Path(BATCH_DIR))
    else:
        run_single(client)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
