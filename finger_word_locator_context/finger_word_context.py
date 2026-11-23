# -*- coding: utf-8 -*-
"""
finger_word_context.py

Version de finger_word_locator.py que, ademas de la palabra senalada, obtiene
las palabras que la acompanan (hasta 3 antes y 3 despues, con las condiciones
de paro de la seccion 5.3) y las muestra en pantalla.

Flujo:
1. Se obtiene una imagen (webcam -> ESPACIO para capturar, o un archivo).
2. Google Vision OCR (document_text_detection) devuelve las palabras en orden
   de lectura (pagina -> bloque -> parrafo -> palabra), con bounding box,
   confianza y el parrafo al que pertenece cada una.
3. MediaPipe Hand Landmarker devuelve la punta del dedo indice (landmark 8).
4. Se elige la palabra mas cercana a la punta del dedo (distancia
   punto-poligono) y se guarda su indice i en el arreglo de palabras.
5. Se extrae la ventana de contexto: hasta 3 palabras hacia atras y 3 hacia
   adelante, deteniendose en cambio de parrafo o fin de oracion (. ? !).
6. Se muestra en pantalla la foto con la palabra senalada (verde), las
   palabras anteriores (naranja) y posteriores (azul), mas un panel con el
   texto. Todo se guarda en output/.

Uso:
    python finger_word_context.py              -> abre la webcam
    python finger_word_context.py foto.jpg     -> usa una imagen existente
"""

import os
import sys
import json
import math
import urllib.request
from functools import lru_cache

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont

from google.cloud import vision as gcv

from context_window import (
    CONTEXT_WORDS_PER_SIDE,
    is_trailing_punct_only,
    extract_context_window,
    window_text_marked,
)


# =========================================================================
# CONFIGURACION
# =========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelo de MediaPipe (se descarga solo si no existe)
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Credenciales de Google Cloud Vision
GOOGLE_CREDENTIALS_PATH = os.path.join(
    BASE_DIR, "credentials", "service_account.json"
)

# Carpeta donde se guardan resultados
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Padding (px) alrededor del recorte de la palabra
CROP_PADDING = 6

# Landmark de la punta del dedo indice en MediaPipe Hands
INDEX_FINGER_TIP = 8

# Confianza minima del OCR para la palabra senalada (RNF02).
# Si ENFORCE_MIN_CONFIDENCE es True, por debajo del minimo el script se
# detiene (equivale al "tono de error" del sistema real). Si es False solo
# muestra una advertencia, util mientras haces pruebas.
MIN_CONFIDENCE = 0.85
ENFORCE_MIN_CONFIDENCE = False

# Tamano maximo (px) de la ventana en pantalla. La imagen guardada no se
# reduce; solo se escala para mostrarla.
MAX_DISPLAY_W = 1100
MAX_DISPLAY_H = 650

# Colores (RGB)
COLOR_TARGET = (0, 200, 0)      # verde: palabra senalada
COLOR_BEFORE = (255, 140, 0)    # naranja: palabras anteriores
COLOR_AFTER = (60, 140, 255)    # azul: palabras posteriores
COLOR_FINGER = (230, 0, 0)      # rojo: punta del dedo
COLOR_OTHER = (180, 180, 180)   # gris: resto de palabras del OCR
COLOR_PANEL_BG = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


# =========================================================================
# UTILIDADES
# =========================================================================

def ensure_hand_model():
    """Descarga el modelo de MediaPipe si no existe localmente."""
    if not os.path.exists(HAND_MODEL_PATH):
        print("Descargando modelo de deteccion de manos...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Modelo descargado correctamente.")


def get_vision_client():
    """Crea el cliente de Google Vision usando el service account local."""
    from google.oauth2 import service_account

    if os.path.exists(GOOGLE_CREDENTIALS_PATH):
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDENTIALS_PATH
        )
        return gcv.ImageAnnotatorClient(credentials=creds)

    # Si no hay archivo local, intenta con la variable de entorno estandar
    return gcv.ImageAnnotatorClient()


def capture_photo_from_webcam():
    """Abre la camara y espera a que el usuario presione ESPACIO para
    capturar una foto. ESC cancela."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo acceder a la camara.")
        sys.exit(1)

    print("Presiona ESPACIO para tomar la foto, ESC para cancelar.")
    frame = None
    while True:
        success, live_frame = cap.read()
        if not success:
            break

        # Se dibuja el texto de ayuda sobre una COPIA, para que no quede
        # impreso en la foto capturada (el OCR lo leeria como texto).
        display = live_frame.copy()
        cv2.putText(
            display, "ESPACIO = capturar | ESC = salir",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        cv2.imshow("Captura - apunta con tu dedo indice", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == 32:  # ESPACIO
            frame = live_frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame


# =========================================================================
# OCR: palabras en orden de lectura
# =========================================================================

def get_ocr_words(image_bgr):
    """Corre Google Vision OCR y devuelve las palabras en ORDEN DE LECTURA
    (pagina -> bloque -> parrafo -> palabra).

    Cada palabra es un dict:
      idx        -> posicion en el arreglo
      text       -> texto de la palabra
      bbox       -> 4 vertices (x, y) en pixeles
      confidence -> confianza del OCR para esa palabra (0 a 1)
      para_id    -> numero de parrafo (unico en toda la imagen)

    Si el OCR entrega un signo suelto (".", ",", ")"...) como palabra
    aparte, se pega a la palabra anterior del mismo parrafo, para que el
    fin de oracion se detecte bien ("house" + "." -> "house.").
    """
    client = get_vision_client()

    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("No se pudo codificar la imagen para OCR.")

    gcv_image = gcv.Image(content=encoded.tobytes())
    response = client.document_text_detection(image=gcv_image)

    if response.error.message:
        raise RuntimeError(f"Error de Vision API: {response.error.message}")

    words = []
    para_id = -1
    if response.full_text_annotation:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    para_id += 1
                    for word in paragraph.words:
                        text = "".join(s.text for s in word.symbols)
                        verts = [(v.x, v.y)
                                 for v in word.bounding_box.vertices]
                        if len(verts) != 4 or not text.strip():
                            continue

                        if (is_trailing_punct_only(text) and words
                                and words[-1]["para_id"] == para_id):
                            words[-1]["text"] += text
                            continue

                        words.append({
                            "idx": len(words),
                            "text": text,
                            "bbox": verts,
                            "confidence": float(word.confidence),
                            "para_id": para_id,
                        })
    return words


# =========================================================================
# MEDIAPIPE: punta del dedo
# =========================================================================

def get_index_fingertip(image_bgr, detector):
    """Corre MediaPipe Hand Landmarker (modo VIDEO, con un solo frame) y
    devuelve las coordenadas en pixeles (x, y) del landmark 8, o None si no
    se detecto ninguna mano.

    Nota: se usa running_mode=VIDEO (igual que en hand_tracking.py) en vez
    de IMAGE porque en la practica resulto mas confiable para fotos
    individuales con poses de mano poco comunes o iluminacion dificil.
    """
    h, w, _ = image_bgr.shape
    print(f"  (debug) tamano de imagen: {w}x{h}")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # timestamp arbitrario, ya que solo procesamos un frame
    result = detector.detect_for_video(mp_image, 1)

    if not result.hand_landmarks:
        print("  (debug) el detector no encontro ninguna mano en este frame.")
        return None

    print(f"  (debug) manos detectadas: {len(result.hand_landmarks)}")

    # Si hay varias manos, se usa la primera detectada
    lm = result.hand_landmarks[0][INDEX_FINGER_TIP]
    return (int(lm.x * w), int(lm.y * h))


# =========================================================================
# SELECCION DE LA PALABRA (distancia punto-poligono)
# =========================================================================

def point_to_segment_distance(p, a, b):
    """Distancia de un punto p al segmento a-b."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_in_polygon(point, polygon):
    """Ray casting para saber si un punto esta dentro de un poligono (bbox)."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def distance_point_to_bbox(point, bbox):
    """Distancia minima de un punto a un bounding box (poligono de 4 vertices).
    Devuelve 0 si el punto esta dentro del bbox."""
    if point_in_polygon(point, bbox):
        return 0.0
    n = len(bbox)
    return min(
        point_to_segment_distance(point, bbox[i], bbox[(i + 1) % n])
        for i in range(n)
    )


def find_nearest_word(fingertip, words):
    """Devuelve (indice, distancia) de la palabra mas cercana a la punta del
    dedo, o None si no hay palabras."""
    if not words:
        return None
    best_idx = None
    best_dist = float("inf")
    for k, w in enumerate(words):
        d = distance_point_to_bbox(fingertip, w["bbox"])
        if d < best_dist:
            best_dist = d
            best_idx = k
    return best_idx, best_dist


def crop_word(image_bgr, bbox, padding=CROP_PADDING):
    """Recorta la region rectangular que envuelve el bounding box, con un
    margen de padding en pixeles."""
    h, w, _ = image_bgr.shape
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)
    return image_bgr[y1:y2, x1:x2]


# =========================================================================
# DIBUJO / PANTALLA
# (se usa PIL para el texto porque cv2.putText no dibuja acentos ni enes)
# =========================================================================

@lru_cache(maxsize=16)
def load_font(size):
    """Carga una fuente TrueType con soporte Unicode; si no encuentra
    ninguna, usa la fuente por defecto de PIL."""
    candidates = [
        "arial.ttf", "segoeui.ttf", "calibri.ttf",          # Windows
        "DejaVuSans.ttf",                                    # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf",      # macOS
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antiguo
        return ImageFont.load_default()


def draw_labels(image_bgr, labels):
    """Dibuja etiquetas de texto con fondo de color sobre la imagen.
    labels: lista de (texto, (x, y), color_fondo_rgb, tamano_fuente)."""
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, (x, y), bg, size in labels:
        font = load_font(size)
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        pad = max(2, size // 6)
        draw.rectangle((l - pad, t - pad, r + pad, b + pad), fill=bg)
        draw.text((x, y), text, font=font, fill=COLOR_TEXT)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def bbox_label_pos(bbox, size, img_h):
    """Posicion (x, y) para una etiqueta sobre el bbox (o debajo si no hay
    espacio arriba)."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x = max(0, min(xs))
    y = min(ys) - size - 8
    if y < 0:
        y = min(img_h - size - 4, max(ys) + 6)
    return (x, y)


def draw_context_image(image_bgr, words, fingertip, window):
    """Copia de la imagen con: todas las palabras del OCR (gris), palabras
    anteriores (naranja, -3 -2 -1), palabra senalada (verde, gruesa),
    palabras posteriores (azul, +1 +2 +3) y la punta del dedo (rojo)."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    t = max(1, round(min(h, w) / 480))   # escala el grosor con el tamano
    fsize = 14 * t

    for wd in words:
        pts = np.array(wd["bbox"], dtype=np.int32)
        cv2.polylines(out, [pts], True, bgr(COLOR_OTHER), t)

    labels = []

    n_before = len(window["before"])
    for k, wd in enumerate(window["before"]):
        pts = np.array(wd["bbox"], dtype=np.int32)
        cv2.polylines(out, [pts], True, bgr(COLOR_BEFORE), 2 * t)
        labels.append((str(k - n_before),
                       bbox_label_pos(wd["bbox"], fsize, h),
                       COLOR_BEFORE, fsize))

    for k, wd in enumerate(window["after"], start=1):
        pts = np.array(wd["bbox"], dtype=np.int32)
        cv2.polylines(out, [pts], True, bgr(COLOR_AFTER), 2 * t)
        labels.append(("+" + str(k),
                       bbox_label_pos(wd["bbox"], fsize, h),
                       COLOR_AFTER, fsize))

    target = window["target"]
    pts = np.array(target["bbox"], dtype=np.int32)
    cv2.polylines(out, [pts], True, bgr(COLOR_TARGET), 3 * t)
    labels.append((target["text"],
                   bbox_label_pos(target["bbox"], fsize + 2 * t, h),
                   COLOR_TARGET, fsize + 2 * t))

    if fingertip is not None:
        cv2.circle(out, fingertip, 6 * t, bgr(COLOR_FINGER), -1)
        labels.append(("dedo", (fingertip[0] + 10 * t, fingertip[1]),
                       COLOR_FINGER, fsize))

    return draw_labels(out, labels)


def wrap_text(draw, text, font, max_width):
    """Divide un texto en lineas que no excedan max_width (en pixeles)."""
    lines, current = [], ""
    for word in text.split(" "):
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def build_info_panel(width, chosen, window, distance):
    """Panel de texto (ancho = width) con la palabra senalada, las palabras
    anteriores y posteriores y la ventana completa."""
    font = load_font(19)
    pad = 12
    line_h = 27

    before = " ".join(w["text"] for w in window["before"]) or "(ninguna)"
    after = " ".join(w["text"] for w in window["after"]) or "(ninguna)"
    rows = [
        ("Palabra señalada:", COLOR_TARGET,
         f"{chosen['text']}   (confianza {chosen['confidence']:.2f}, "
         f"distancia {distance:.1f} px)"),
        (f"Antes ({len(window['before'])}):", COLOR_BEFORE, before),
        (f"Después ({len(window['after'])}):", COLOR_AFTER, after),
        ("Ventana:", COLOR_TEXT, window_text_marked(window)),
        ("Posición en la ventana:", COLOR_TEXT, str(window["target_pos"])),
    ]

    # medir
    meas = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_w = max(meas.textlength(lbl, font=font) for lbl, _, _ in rows) + 12
    text_w = max(50, width - 2 * pad - label_w)
    wrapped = [wrap_text(meas, txt, font, text_w) for _, _, txt in rows]
    height = 2 * pad + sum(len(w) for w in wrapped) * line_h

    img = Image.new("RGB", (width, height), COLOR_PANEL_BG)
    draw = ImageDraw.Draw(img)
    y = pad
    for (lbl, color, _), lines in zip(rows, wrapped):
        draw.text((pad, y), lbl, font=font, fill=color)
        for ln in lines:
            draw.text((pad + label_w, y), ln, font=font, fill=COLOR_TEXT)
            y += line_h
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def compose_display(annotated, panel_fn):
    """Escala la imagen anotada para mostrarla y le pega debajo el panel de
    texto. panel_fn(ancho) -> imagen BGR del panel."""
    h, w = annotated.shape[:2]
    scale = min(1.0, MAX_DISPLAY_W / w, MAX_DISPLAY_H / h)
    if scale < 1.0:
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    panel = panel_fn(annotated.shape[1])
    return np.vstack([annotated, panel])


# =========================================================================
# MAIN
# =========================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_hand_model()

    # --- 1. Obtener la imagen (webcam o archivo dado por argumento) ---
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            sys.exit(1)
    else:
        image = capture_photo_from_webcam()
        if image is None:
            print("No se capturo ninguna imagen.")
            sys.exit(1)
        captured_path = os.path.join(OUTPUT_DIR, "captura.jpg")
        cv2.imwrite(captured_path, image)
        print(f"Foto guardada en: {captured_path}")

    # --- 2. OCR: palabras en orden de lectura ---
    print("Corriendo OCR (Google Vision)...")
    words = get_ocr_words(image)
    print(f"Se detectaron {len(words)} palabras.")

    # --- 3. MediaPipe: detectar punta del dedo indice ---
    print("Detectando mano...")
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    fingertip = get_index_fingertip(image, detector)

    if fingertip is None:
        print("No se detecto ninguna mano en la imagen. Abortando.")
        sys.exit(1)
    print(f"Punta del indice detectada en: {fingertip}")

    # --- 4. Palabra mas cercana (indice i dentro del arreglo) ---
    nearest = find_nearest_word(fingertip, words)
    if nearest is None:
        print("No se detecto texto en la imagen. Abortando.")
        sys.exit(1)

    i, distance = nearest
    chosen = words[i]
    print(f"Palabra mas cercana: '{chosen['text']}' "
          f"(indice {i}, distancia: {distance:.1f}px, "
          f"confianza: {chosen['confidence']:.2f})")

    if chosen["confidence"] < MIN_CONFIDENCE:
        print(f"  AVISO: confianza {chosen['confidence']:.2f} < "
              f"{MIN_CONFIDENCE} (RNF02).")
        if ENFORCE_MIN_CONFIDENCE:
            print("Abortando (ENFORCE_MIN_CONFIDENCE = True).")
            sys.exit(1)

    # --- 5. Ventana de contexto: hasta N palabras por lado ---
    window = extract_context_window(words, i, CONTEXT_WORDS_PER_SIDE)
    print("--- Ventana de contexto ---")
    print("  Antes   :", " ".join(w["text"] for w in window["before"])
          or "(ninguna)")
    print("  Palabra :", chosen["text"])
    print("  Despues :", " ".join(w["text"] for w in window["after"])
          or "(ninguna)")
    print("  Ventana :", window_text_marked(window))
    print("  Posicion de la palabra en la ventana:", window["target_pos"])

    # --- 6. Guardar resultados ---
    crop = crop_word(image, chosen["bbox"])
    crop_path = os.path.join(OUTPUT_DIR, "palabra_recortada.jpg")
    cv2.imwrite(crop_path, crop)

    annotated = draw_context_image(image, words, fingertip, window)
    debug_path = os.path.join(OUTPUT_DIR, "debug.jpg")
    cv2.imwrite(debug_path, annotated)

    display = compose_display(
        annotated,
        lambda width: build_info_panel(width, chosen, window, distance),
    )
    context_path = os.path.join(OUTPUT_DIR, "contexto.jpg")
    cv2.imwrite(context_path, display)
    print(f"Imagenes guardadas en: {OUTPUT_DIR}")

    result_json = {
        # campos de la version anterior
        "word": chosen["text"],
        "bbox": chosen["bbox"],
        "fingertip": fingertip,
        "distance_px": distance,
        # campos nuevos
        "confidence": chosen["confidence"],
        "index": i,
        "paragraph_id": chosen["para_id"],
        "context": {
            "words_per_side": CONTEXT_WORDS_PER_SIDE,
            "before": [w["text"] for w in window["before"]],
            "after": [w["text"] for w in window["after"]],
            "window_text": window["text"],
            "target_pos": window["target_pos"],
            "window_words": [
                {"text": w["text"], "bbox": w["bbox"]}
                for w in (window["before"] + [window["target"]]
                          + window["after"])
            ],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "resultado.json"), "w",
              encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    # --- 7. Mostrar en pantalla ---
    cv2.imshow("Resultado (presiona cualquier tecla para cerrar)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
