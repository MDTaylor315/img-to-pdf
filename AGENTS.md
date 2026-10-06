# Notas para el backend — Endpoint `/api/gd/imagenes_pdf`

Este repositorio contiene el motor Python (`src/pipeline.py`) que el backend debe consumir para el endpoint `POST /api/gd/imagenes_pdf`.

## Contratos

### Request

- **Multipart/form-data** con los siguientes campos:
  - `db` (string, obligatorio)
  - `id_user` (string/integer, obligatorio)
  - `fotos` (files, obligatorio, puede repetirse; en Postman usar la opción "Send with the same key" para subir varios archivos con el mismo campo)

### Response exitosa

- Status `2xx`
- `Content-Type: application/pdf`
- Body: bytes del PDF comenzando por `%PDF-`

### Response de error

- Status `400` (u otro 4xx/5xx apropiado)
- `Content-Type: application/json`
- Body:

```json
{
  "success": false,
  "message": "..."
}
```

## Integración mínima en FastAPI

```python
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from src.pipeline import (
    ErrorValidacionEntrada,
    convertir_fotos_a_pdf_endpoint,
)

app = FastAPI()

@app.post("/api/gd/imagenes_pdf")
async def imagenes_pdf(
    db: str = Form(...),
    id_user: str = Form(...),
    fotos: list[UploadFile] = File(...),
):
    try:
        pdf_bytes = convertir_fotos_a_pdf_endpoint(db, id_user, fotos)
    except ErrorValidacionEntrada as exc:
        raise HTTPException(status_code=exc.codigo, detail={"success": False, "message": exc.mensaje})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"success": False, "message": str(exc)})

    return Response(content=pdf_bytes, media_type="application/pdf")
```

## Integración mínima en Flask

```python
from flask import Flask, request, Response, jsonify
from src.pipeline import (
    ErrorValidacionEntrada,
    convertir_fotos_a_pdf_endpoint,
)

app = Flask(__name__)

@app.route("/api/gd/imagenes_pdf", methods=["POST"])
def imagenes_pdf():
    db = request.form.get("db")
    id_user = request.form.get("id_user")
    fotos = request.files.getlist("fotos")

    try:
        pdf_bytes = convertir_fotos_a_pdf_endpoint(db, id_user, fotos)
    except ErrorValidacionEntrada as exc:
        return jsonify({"success": False, "message": exc.mensaje}), exc.codigo
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500

    return Response(pdf_bytes, mimetype="application/pdf")
```

## Configuración de timeouts y tamaño de multipart

El procesamiento de 15-20 fotos de alta resolución puede tardar decenas de segundos. Configurar **≥180 s** en toda la cadena:

### Gunicorn

```bash
gunicorn main:app \
  --timeout 180 \
  --workers 2 \
  --threads 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --keepalive 5 \
  --limit-request-line 8190
```

### Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 180
```

> Nota: Uvicorn por sí solo no expone `--timeout` para workers; usar Gunicorn con `UvicornWorker` y `--timeout 180` para controlar el tiempo máximo de una petición.

### Nginx (proxy inverso)

```nginx
location /api/gd/imagenes_pdf {
    proxy_pass http://backend;
    proxy_read_timeout 180s;
    proxy_send_timeout 180s;
    proxy_connect_timeout 180s;
    client_max_body_size 100m;
    proxy_request_buffering off;   # opcional: permite streaming
}
```

### FastAPI / Starlette — límite de multipart

```python
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

# En el router o middleware, asegurar que los límites sean suficientes:
# python-multipart y Starlette usan el tamaño total del body (100 MB).
```

### Flask — límite de multipart

```python
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
```

## Limpieza de PDFs huérfanos (opcional)

Si el backend guarda PDFs generados temporalmente (por ejemplo, para un flujo `job_id` + polling), programar una tarea periódica que borre archivos no descargados después de 24 horas.

Ejemplo con APScheduler:

```python
from apscheduler.schedulers.background import BackgroundScheduler
import os

def limpiar_pdfs_huerfanos(carpeta: str, max_edad_horas: int = 24):
    ahora = time.time()
    for nombre in os.listdir(carpeta):
        ruta = os.path.join(carpeta, nombre)
        if os.path.isfile(ruta) and (ahora - os.path.getmtime(ruta)) > max_edad_horas * 3600:
            os.remove(ruta)

scheduler = BackgroundScheduler()
scheduler.add_job(limpiar_pdfs_huerfanos, "interval", hours=1, args=["/tmp/pdf_jobs"])
scheduler.start()
```

## Rendimiento medido (referencia)

Medido en el entorno local con las 5 fotos de prueba de `input/` (~2 MB cada una, imagen original de ~4096 px de ancho):

| Escenario | Tiempo total (5 fotos) | Tiempo/foto aprox. | PDF final |
|-----------|------------------------|--------------------|-----------|
| Original (config por defecto) | ~94 s | ~19 s | 2.3 MB (~460 KB/página) |
| Con optimización del filtro de iluminación + pre-downscale | ~48 s | ~10 s | 2.2 MB (~440 KB/página) |
| Con todo lo anterior + procesamiento paralelo (2 hilos) | **~32 s** | **~6.4 s** | **1.5 MB (~290 KB/página)** |
| Con todo lo anterior + procesamiento paralelo (4 hilos) | ~25 s | ~5 s | 1.5 MB (~290 KB/página) |
| Sin detector DL ni orientación automática | ~13 s | ~2.5 s | similar |

Con el procesamiento paralelo por defecto (2 hilos), 18 fotos extrapolarían aproximadamente **~115 s**, bien por debajo del límite de 180 s.

Recomendaciones:

1. Configurar el timeout a **180 s** en toda la cadena.
2. Mantener el procesamiento paralelo con **2 hilos** (`MAX_WORKERS_PROCESAMIENTO = 2`). Es el default y es un buen balance para servidores con poca RAM o alta carga.
3. Si el servidor tiene **mucha RAM libre y poca concurrencia**, podés subir a `4` para bajar a ~90 s con 18 fotos.
4. Si se siguen viendo cierres de conexión, desactivar en `src/config.py` las funciones más pesadas:
   ```python
   USAR_DETECTOR_DL = False
   USAR_MODELO_ORIENTACION_ONNX = False
   AUTO_ORIENTAR_TEXTO_DEFECTO = False
   USAR_AUTO_CROP_DEFECTO = False
   ```
5. Si aun así se acerca a 180 s, implementar **job_id + polling** (ver sección anterior).

## Dependencias

Para HEIC/HEIF instalar la versión compatible con el ambiente del backend (por ejemplo, `0.12.0`):

```bash
pip install pillow-heif==0.12.0
```

El `requirements.txt` actual utiliza `pillow-heif==0.12.0`. Si se instala, `pipeline.py` registra el opener automáticamente para que Pillow pueda abrir archivos `.heic`/`.heif`.

Nota: en Windows con Python 3.13 puede no haber wheel pre-compilado para `0.12.0`; en el ambiente de producción del backend debe verificar que la versión se instale correctamente.
