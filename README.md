# MontanoImagen - Procesador de Imágenes a PDF HD (CamScanner Style)

Sistema modular en Python de alto rendimiento para procesar imágenes capturadas desde celulares (facturas, guías de remisión, vástagos de recepción) y convertirlas en **PDFs con calidad HD estilo escáner profesional**.

Diseñado específicamente para consumo cero de archivos basura en disco duro, bajo uso de memoria RAM e integración nativa en servidores web y sistemas ERP como **Odoo**.

## 🚀 Características Principales

* **Filtro Mágico HD (CamScanner Style):** 
  * Blanqueamiento de fondo uniforme (RGB 255) eliminando sombras de celular, manos o luz ambiente tenue.
  * Ecualización de contraste local CLAHE + curva LUT no lineal para tinta, firmas a mano y sellos oscuros.
  * Nitidez y enfoque adaptativo para texto claro y códigos de barras.
* **Auto-Crop Seguro:** Detección conservadora de bordes de papel de 4 esquinas sin recortar tablas ni cabeceras por error.
* **Optimizado para Servidores (Zero-Hog):**
  * Límite de hilos de CPU por proceso (`NUM_HILOS_OPENCV = 1`) para evitar picos en horas punta.
  * Análisis de contornos en miniaturas a 800px (<10ms).
  * Procesamiento por lotes *streamed* página por página con liberación explícita de memoria RAM (`gc.collect()`).
* **Integración Nactiva con Odoo / APIs en RAM:** Procesa buffers de imágenes en memoria Base64/Bytes sin tocar el disco duro.

---

## 🛠️ Instalación

Requisitos previos: Python 3.8+

```bash
pip install -r requirements.txt
```

### Dependencias (`requirements.txt`)
* `opencv-python-headless`
* `numpy`
* `Pillow`
* `img2pdf`

---

## 💻 Uso Básico

### 1. Procesar fotos locales de prueba
Coloca tus imágenes (`.jpg`, `.png`) en la carpeta `input/` y ejecuta:

```bash
python src/pipeline.py
```

El PDF resultante se generará en la carpeta `output/`.

---

## ⚙️ Configuración (`src/config.py`)

Puedes ajustar los parámetros de rendimiento y calidad en `src/config.py`:

```python
# Hilos de CPU por proceso (1 para servidores web concurrentes)
NUM_HILOS_OPENCV = 1

# Ancho máximo de la imagen en píxeles (2480px = ancho A4 a 300 DPI)
MAX_ANCHO_IMAGEN = 2480

# Dimensión máxima de trabajo para OpenCV (3508px = alto A4 a 300 DPI)
MAX_DIMENSION_IMAGEN = 3508

# Calidad de compresión JPEG (1 a 100) y DPI incrustado
CALIDAD_JPEG = 85
JPEG_DPI = (300, 300)

# Liberar memoria RAM automáticamente tras cada página
LIMPIAR_RAM_POR_PAGINA = True

# Límites del endpoint
MAX_CANTIDAD_FOTOS = 50
MAX_BYTES_POR_FOTO = 25 * 1024 * 1024   # 25 MB
MAX_BYTES_TOTALES = 100 * 1024 * 1024   # 100 MB
FORMATOS_IMAGEN_PERMITIDOS = {"JPEG", "PNG", "WEBP", "HEIC", "HEIF"}
```

---

## 🏢 Integración en Odoo

Para integrar en un módulo personalizado de Odoo leyendo imágenes de `ir.attachment` o campos `fields.Binary`:

```python
import base64
import img2pdf
from src.pipeline import procesar_imagen_a_bytes

def convertir_adjuntos_odoo_a_pdf(registros_adjuntos_odoo):
    buffers_jpeg_procesados = []

    for adjunto in registros_adjuntos_odoo:
        bytes_foto_original = base64.b64decode(adjunto.datas)
        bytes_jpeg_limpio = procesar_imagen_a_bytes(bytes_foto_original)
        buffers_jpeg_procesados.append(bytes_jpeg_limpio)

    pdf_bytes = img2pdf.convert(buffers_jpeg_procesados)
    return base64.b64encode(pdf_bytes)
```

---

## 🌐 Integración como endpoint web (`/api/gd/imagenes_pdf`)

El motor expone helpers para validar el multipart y generar el PDF. El backend debe montar el endpoint HTTP. Ver `AGENTS.md` para:

- Ejemplos de integración en **FastAPI** y **Flask**.
- Configuración de **timeouts** (Gunicorn/Uvicorn/Nginx) a ≥180 s.
- Límite de **multipart a 100 MB**.
- Contrato de errores `{"success": false, "message": "..."}`.
- Limpieza periódica de PDFs huérfanos si se decide usar `job_id` + polling.
