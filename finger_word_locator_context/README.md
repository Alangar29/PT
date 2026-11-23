# finger_word_locator_context

Version de `finger_word_locator` que, ademas de la palabra senalada con el
dedo, obtiene **las palabras que la acompanan** (hasta 3 antes y 3 despues) y
las **muestra en pantalla** al tomar la foto. Sigue la seccion 5.3 del reporte.

## Archivos

- `finger_word_context.py` — script principal (camara, OCR, MediaPipe, pantalla).
- `context_window.py` — logica pura de la ventana de contexto (sin dependencias).
- `test_ventana_contexto.py` — pruebas de la ventana, sin camara ni APIs.
- `requirements.txt` — dependencias (se agrego `pillow` para dibujar acentos).

## Instalacion

Copia esta carpeta junto a `finger_word_locator` (o dentro de `Test/`) y:

1. Crea el entorno virtual e instala dependencias:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copia tus credenciales de Google Cloud a
   `credentials/service_account.json` (carpeta nueva, junto al script).
3. (Opcional) copia `hand_landmarker.task` de `finger_word_locator` para no
   volver a descargarlo.

## Uso

```powershell
python finger_word_context.py            # webcam: ESPACIO captura, ESC sale
python finger_word_context.py foto.jpg   # con una foto existente
```

Pruebas de la logica (no usa camara ni internet):
```powershell
python test_ventana_contexto.py
```

## Que se muestra

Una ventana con la foto y un panel de texto:

- **Verde**: palabra senalada.
- **Naranja** (-3, -2, -1): palabras anteriores.
- **Azul** (+1, +2, +3): palabras posteriores.
- **Gris**: el resto de palabras detectadas por el OCR. **Rojo**: punta del dedo.
- Panel: palabra senalada, antes, despues, ventana completa
  (`decided to break [up] and sell the`) y posicion de la palabra en ella.

## Reglas de la ventana (N = 3 por lado)

- **Hacia atras**: se detiene antes de cruzar un cambio de parrafo, o si la
  palabra anterior termina en `.` `?` `!`.
- **Hacia adelante**: se detiene si la palabra actual termina en `.` `?` `!`,
  o si la siguiente es de otro parrafo.
- La palabra senalada siempre se incluye; las comas no detienen el recorrido.
- Se cambia `CONTEXT_WORDS_PER_SIDE` en `context_window.py` para variar N.

## Salida (`output/`)

- `contexto.jpg` — lo que ves en pantalla (imagen + panel).
- `debug.jpg` — imagen anotada a tamano original.
- `palabra_recortada.jpg` — recorte de la palabra senalada.
- `resultado.json` — mantiene `word`, `bbox`, `fingertip`, `distance_px` y
  agrega `confidence`, `index`, `paragraph_id` y `context`
  (`before`, `after`, `window_text`, `target_pos`, `window_words`).
  `window_text` y `target_pos` son lo que se enviaria al CORE IA.

## Notas y limitaciones

- El orden de las palabras es el de `full_text_annotation` de Vision (pagina,
  bloque, parrafo, palabra), no el de la lista plana.
- Si el OCR separa un signo suelto (`.`, `,`, `)`), se pega a la palabra
  anterior para detectar bien el fin de oracion.
- Abreviaturas con punto (`Dr.`, `e.g.`) cuentan como fin de oracion.
- Palabras partidas por guion al final de linea (`trans-` / `lation`) todavia
  no se unen.
- `MIN_CONFIDENCE = 0.85`: con `ENFORCE_MIN_CONFIDENCE = False` solo avisa; en
  `True` se detiene como lo haria el tono de error del sistema.
- Se corrigio un detalle de `capture_photo_from_webcam`: el texto
  "ESPACIO = capturar" se dibujaba sobre el mismo frame que se guardaba, y el
  OCR lo leia como parte del texto. Ahora se dibuja sobre una copia.
