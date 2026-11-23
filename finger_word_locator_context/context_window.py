# -*- coding: utf-8 -*-
"""
context_window.py

Logica pura (sin camara, sin MediaPipe, sin Google Vision) para obtener la
ventana de contexto alrededor de la palabra senalada.

Regla (seccion 5.3 del reporte):
- A partir de la palabra senalada (indice i) se toman hasta N palabras hacia
  atras y hasta N hacia adelante (N = 3 por defecto).
- Hacia atras: se detiene antes de cruzar un cambio de parrafo, o si la
  palabra anterior termina en signo de fin de oracion (. ? !).
- Hacia adelante: se detiene si la palabra actual termina en signo de fin de
  oracion, o si la siguiente pertenece a otro parrafo.
- La palabra senalada siempre se incluye. Las comas y otros signos
  intermedios NO detienen el recorrido.

Cada palabra es un dict con al menos: {"text": str, "para_id": int}.
Las palabras deben venir en orden de lectura (como las entrega Google Vision
en full_text_annotation: pagina -> bloque -> parrafo -> palabra).
"""

CONTEXT_WORDS_PER_SIDE = 3

# Signos que cierran una oracion
SENTENCE_END_CHARS = ".?!…"
# Signos de cierre que pueden ir despues del punto: ej. house.") o "stop!"
CLOSING_CHARS = ")]}\"'”’»"
# Tokens formados solo por estos signos se pegan a la palabra anterior
TRAILING_PUNCT = set(".,;:!?…)]}\"'”’»")


def ends_sentence(text):
    """True si el texto termina en signo de fin de oracion (., ?, !, …),
    ignorando comillas o parentesis de cierre al final: 'house.' 'stop!"'"""
    core = text.rstrip(CLOSING_CHARS)
    return len(core) > 0 and core[-1] in SENTENCE_END_CHARS


def is_trailing_punct_only(text):
    """True si el token es solo puntuacion de cierre (ej. '.', ',', ')').
    Sirve para pegarlo a la palabra anterior cuando el OCR lo separa."""
    return len(text) > 0 and all(ch in TRAILING_PUNCT for ch in text)


def extract_context_window(words, i, n=CONTEXT_WORDS_PER_SIDE):
    """Devuelve la ventana de contexto de words[i].

    Retorna un dict:
      start, end   -> indices (inclusivos) de la ventana dentro de `words`
      target_pos   -> posicion de la palabra senalada dentro de la ventana
      before       -> lista de palabras anteriores (en orden de lectura)
      target       -> la palabra senalada
      after        -> lista de palabras posteriores
      text         -> frase de contexto completa
    """
    if not (0 <= i < len(words)):
        raise IndexError("Indice de palabra fuera de rango")

    para = words[i]["para_id"]

    # --- hacia atras ---
    left = i
    while (
        left > 0
        and (i - left) < n
        and words[left - 1]["para_id"] == para
        and not ends_sentence(words[left - 1]["text"])
    ):
        left -= 1

    # --- hacia adelante ---
    right = i
    while (
        right < len(words) - 1
        and (right - i) < n
        and words[right + 1]["para_id"] == para
        and not ends_sentence(words[right]["text"])
    ):
        right += 1

    return {
        "start": left,
        "end": right,
        "target_pos": i - left,
        "before": words[left:i],
        "target": words[i],
        "after": words[i + 1:right + 1],
        "text": " ".join(w["text"] for w in words[left:right + 1]),
    }


def window_text_marked(window):
    """Frase de contexto con la palabra senalada entre corchetes:
    'decided to break [up] and sell the'"""
    parts = [w["text"] for w in window["before"]]
    parts.append("[" + window["target"]["text"] + "]")
    parts += [w["text"] for w in window["after"]]
    return " ".join(parts)
