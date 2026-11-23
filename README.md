# Finger Word Locator con Ventana de Contexto

Aplicación que usa la cámara o una imagen para:

1. Detectar la punta del dedo índice con MediaPipe.
2. Ejecutar OCR con Google Cloud Vision.
3. Seleccionar la palabra más cercana al dedo.
4. Extraer hasta tres palabras antes y tres después.
5. Guardar la imagen anotada y los resultados en JSON.

## Requisitos

- Windows
- Python 3.11 o 3.12 recomendado
- Cámara web, si se ejecutará con captura en vivo
- Cuenta y proyecto de Google Cloud con Vision API habilitada
- Archivo JSON de credenciales de una cuenta de servicio

Las dependencias Python son:

```text
opencv-python
numpy
mediapipe
google-cloud-vision
pillow
```

## Instalación en Windows

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\finger_word_locator_context\requirements.txt
```

Si PowerShell bloquea la activación del entorno virtual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

En Visual Studio Code también puede seleccionarse el intérprete:

1. Abrir la paleta con `Ctrl+Shift+P`.
2. Ejecutar `Python: Select Interpreter`.
3. Elegir `.venv\Scripts\python.exe`.

## Configuración de Google Cloud

Colocar las credenciales en:

```text
finger_word_locator_context\credentials\service_account.json
```

Ejemplo de creación de carpeta desde la raíz:

```powershell
New-Item -ItemType Directory -Path .\finger_word_locator_context\credentials
```

El proyecto puede autenticar Google Vision mediante:

- El archivo `credentials/service_account.json`, o
- La variable de entorno `GOOGLE_APPLICATION_CREDENTIALS`.

Ejemplo con variable de entorno:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\ruta\service_account.json"
```

## Ejecución

Entrar al directorio del programa:

```powershell
cd .\finger_word_locator_context
```

Capturar una imagen desde la cámara:

```powershell
.\..\.venv\Scripts\python.exe .\finger_word_context.py
```

Al abrir la cámara:

- `ESPACIO`: tomar la fotografía.
- `ESC`: cerrar sin procesar.

Procesar una imagen existente:

```powershell
.\..\.venv\Scripts\python.exe .\finger_word_context.py .\ruta\imagen.jpg
```

## Pruebas de la ventana de contexto

Estas pruebas no usan cámara ni APIs externas:

```powershell
cd .\finger_word_locator_context
.\..\.venv\Scripts\python.exe .\test_ventana_contexto.py
```

Resultado esperado: `Ran 10 tests ... OK`.

## Cómo funciona la ventana de contexto

La palabra señalada siempre se incluye. Después se toman hasta tres palabras por lado, salvo que se encuentre:

- Un cambio de párrafo.
- Un punto o cierre de oración (`.`, `?`, `!`, `…`).
- El inicio o final de la lista de palabras.

Comas y signos intermedios no detienen la búsqueda.

El número máximo se puede cambiar con `CONTEXT_WORDS_PER_SIDE` en `context_window.py`.

## Archivos principales

```text
finger_word_locator_context/
├── finger_word_context.py       Programa principal
├── context_window.py            Lógica de ventana de contexto
├── test_ventana_contexto.py     Pruebas unitarias
├── hand_landmarker.task         Modelo local de MediaPipe
├── requirements.txt             Dependencias Python
└── output/                      Archivos generados
```

## Resultados generados

Dentro de `finger_word_locator_context\output`:

- `captura.jpg`: imagen obtenida con la cámara.
- `contexto.jpg`: imagen anotada y panel de información.
- `debug.jpg`: anotaciones sobre la imagen original.
- `palabra_recortada.jpg`: recorte de la palabra señalada.
- `resultado.json`: palabra, posición, confianza, dedo, distancia y contexto.

Ejemplo resumido de la estructura guardada:

```json
{
  "word": "palabra",
  "confidence": 0.98,
  "index": 12,
  "paragraph_id": 2,
  "context": {
    "before": [],
    "after": [],
    "window_text": "palabra",
    "target_pos": 0,
    "window_words": []
  }
}
```

## Solución de problemas

### La cámara no abre

- Verificar permisos de cámara en Windows.
- Cerrar otras aplicaciones que estén usando la cámara.
- Comprobar que el script pueda acceder a `cv2.VideoCapture(0)`.

### Error de Google Vision

- Confirmar que Vision API esté habilitada en Google Cloud.
- Verificar que el JSON de credenciales exista en la ruta indicada.
- Comprobar que la cuenta de servicio tenga permisos de Vision.

### MediaPipe no encuentra el modelo

`finger_word_context.py` descarga `hand_landmarker.task` automáticamente si no existe. También debe estar disponible una conexión a Internet durante la primera ejecución.

### Conflicto entre paquetes

Para usar solamente las dependencias del proyecto:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
```

`npm install opencv-python` no es necesario: OpenCV es una dependencia de Python y se instala con `pip`, no con npm.
