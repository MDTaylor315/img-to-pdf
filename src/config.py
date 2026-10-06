import os

# ==============================================================================
# CONFIGURACIÓN DEL SISTEMA DE ESCANEO DE IMÁGENES A PDF (MontanoImagen)
# Modifica estos valores para controlar la calidad y la carga del servidor.
# ==============================================================================

# --- RENDIMIENTO Y CONTROL DE CPU EN HORA PUNTA ---
# Número de hilos de CPU que puede usar OpenCV por proceso.
# RECOMENDACIÓN EN HORA PUNTA: Setear a 1 para evitar saturación de vCPUs concurrentes.
NUM_HILOS_OPENCV = 1

# Ancho máximo en píxeles para redimensionar la imagen HD antes de empaquetarla al PDF.
# 2480 px = ancho de una hoja A4 a 300 dpi.
# TIP HORA PUNTA: Si el servidor recibe muchas peticiones simultáneas,
# puedes reducirlo a 1800 px para acelerar el procesamiento un 40% adicional.
MAX_ANCHO_IMAGEN = 2480

# Dimensión máxima (ancho o alto) permitida durante el procesamiento OpenCV.
# 3508 px = alto de una hoja A4 a 300 dpi. Sirve para evitar costos excesivos
# con fotos de muy alta resolución antes del re-encodeo final con Pillow.
MAX_DIMENSION_IMAGEN = 3508

# Dimensión máxima de la miniatura para análisis rápido de contornos.
MAX_DIM_MINIATURA_ANALISIS = 800


# --- CONTROL DE MEMORIA RAM ---
# Fuerza la liberación explícita de memoria RAM (Garbage Collection) tras procesar cada foto.
# Mantiene el consumo de RAM plano (~15-25 MB por usuario) sin importar si son 10 o 200 fotos.
LIMPIAR_RAM_POR_PAGINA = True


# --- PARAMETROS DE CALIDAD Y COMPRESIÓN ---
# Calidad de compresión JPEG guardado en RAM (1 a 100).
# 85: Balance perfecto de nitidez en firmas/texto con bajo peso (~400-700 KB por hoja A4).
CALIDAD_JPEG = 85

# Resolución lógica incrustada en el JPEG/PDF para impresión a 300 dpi.
JPEG_DPI = (300, 300)

# Modo de procesamiento por defecto:
# - "magico": Escáner HD profesional (fondo blanco pulcro, texto y sellos oscuros).
# - "otsu": Blanco y Negro puro (1-bit).
MODO_PROCESAMIENTO_DEFECTO = "magico"

# Activar o desactivar recorte automático de perspectiva por defecto.
# RECOMENDACIÓN: False para no arriesgar recorte de cabeceras en fotos cerradas.
USAR_AUTO_CROP_DEFECTO = True

# Umbral mínimo de cobertura de papel para justificar auto-crop (0.20 = 20% de la foto).
PORCENTAJE_MIN_COBERTURA_PAPEL = 0.20

# Umbral máximo de cobertura de papel (0.98). Permite recortar el papel de la foto
# descartando fondos, mesas y contornos no deseados alrededor de la hoja.
PORCENTAJE_MAX_COBERTURA_PAPEL = 0.98


# Activar o desactivar auto-orientación por defecto.
# Corrige fotos tomadas con el celular mirando a una mesa cuando el giroscopio se confunde (90°/270°/180°).
AUTO_ORIENTAR_TEXTO_DEFECTO = True

# --- CLASIFICACION DE ORIENTACION INTELIGENTE (ONNX) ---
# Modelo ONNX ultraligero (~6 MB) evaluado con OpenCV DNN (cv2.dnn) en ~7ms.
# Detecta y corrige con precision rotaciones de 0°, 90°, 180° y 270°.
USAR_MODELO_ORIENTACION_ONNX = True
MODELO_ORIENTACION_PATH = os.path.join(os.path.dirname(__file__), "models", "rapid_orientation.onnx")


# --- DETECTOR DE ESQUINAS DE DOCUMENTO VIA DEEP LEARNING (ONNX) ---
# Pipeline de 2 modelos que reemplaza la heuristica OpenCV para recorte de hojas:
#   1. yolo_doc_detector.onnx  (~7 MB)  - Detecta el bounding-box del documento
#   2. lcnet_doc_corners.onnx  (~15 MB) - Regresa las 4 esquinas exactas via heatmaps
# Precision en hojas apiladas: >90% vs ~50-60% del OpenCV heuristico.
# Tiempo adicional en CPU: ~25-45ms por foto (negligible para el usuario).
# Requiere: pip install onnxruntime
# Poner en False para deshabilitar y usar solo la heuristica OpenCV (Tiers 1 y 2).
USAR_DETECTOR_DL = True
MODELO_YOLO_DOC_PATH  = os.path.join(os.path.dirname(__file__), "models", "yolo_doc_detector.onnx")
MODELO_LCNET_DOC_PATH = os.path.join(os.path.dirname(__file__), "models", "lcnet_doc_corners.onnx")

# Umbral de corte anticipado: si la primera orientación evaluada ya da un resultado
# con este score y cobertura de área, se evitan las otras 3 rotaciones (ahorra hasta
# ~75% del costo de CPU del detector DL en el caso común sin cambiar el resultado).
# Bajar estos valores = menos CPU, más confianza en la primera orientación.
# Subirlos = más CPU, más robustez para fotos muy ambiguas (hojas inclinadas, fondos confusos).
SCORE_CORTE_ANTICIPADO_DETECTOR = 7.2
AREA_CORTE_ANTICIPADO_DETECTOR = 0.45


# --- AJUSTE FINO DE IMAGEN (FILTRO MÁGICO) ---
# Límite de corte para ecualización CLAHE (1.0 a 3.0).
CLIP_LIMIT_CLAHE = 1.5

# Fuerza del filtro de nitidez / unsharp mask (1.0 a 2.0).
FUERZA_NITIDEZ = 1.2


# --- VALIDACIONES DE ENTRADA DEL ENDPOINT ---
# Límites para evitar que una petición consuma demasiada memoria o CPU.
MAX_CANTIDAD_FOTOS = 50
MAX_BYTES_POR_FOTO = 25 * 1024 * 1024
MAX_BYTES_TOTALES = 100 * 1024 * 1024
MAX_PIXELES_POR_FOTO = 25_000_000

# Hilos de procesamiento paralelo por request (1 = secuencial).
# Cada hilo procesa una foto completa. Aumenta el pico de RAM y CPU,
# pero reduce el wall-clock de un lote. Ajustar según vCPUs/RAM del servidor.
# Default MÍNIMO consumo de recursos (1 = secuencial, nunca 2 fotos a la vez).
# Si el servidor tiene vCPUs/RAM de sobra y la prioridad es velocidad, subir a 2 o más.
MAX_WORKERS_PROCESAMIENTO = 1

# --- TOPE GLOBAL DE CPU POR PROCESO (no por usuario, no por request) ---
# A diferencia de MAX_WORKERS_PROCESAMIENTO (que limita las fotos en paralelo
# DENTRO de una sola request), esto limita cuántas fotos se procesan al mismo
# tiempo en TODO el proceso worker, sin importar cuántos usuarios/requests
# concurrentes lleguen. Si ya hay este número de fotos procesándose, las demás
# esperan su turno en una cola en vez de competir todas a la vez por el CPU.
# Recomendado: dejarlo en 1 para un tope de CPU fijo y predecible por worker.
# IMPORTANTE: es un límite POR PROCESO. Si el servidor corre varios workers
# (ej. `gunicorn --workers 4`), el tope real de CPU del servidor es
# MAX_PROCESAMIENTO_CONCURRENTE_GLOBAL * cantidad_de_workers. Para un único
# tope real en todo el servidor, correr un solo worker o combinar esto con
# límites de SO (cgroups, `docker --cpus`, `CPUQuota` de systemd).
MAX_PROCESAMIENTO_CONCURRENTE_GLOBAL = 1

# Tiempo máximo (segundos) que una foto puede esperar en la cola del límite
# global antes de abortar con un error claro al cliente, en vez de quedarse
# "pegada" indefinidamente si llega una avalancha de requests.
TIMEOUT_ESPERA_CPU_SEGUNDOS = 150

# Formatos que el pipeline acepta. El endpoint debe aceptar por extensión y delegar
# la validación de contenido a Pillow/OpenCV. HEIC/HEIF requieren pillow-heif instalado.
FORMATOS_IMAGEN_PERMITIDOS = frozenset({"JPEG", "PNG", "WEBP", "HEIC", "HEIF"})
