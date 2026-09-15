import os
import io
import gc
import cv2
import numpy as np
import img2pdf
from PIL import Image, ImageOps

# Cargar constantes de configuración centralizadas
import config

# Aplicar límite de hilos para hora punta
cv2.setNumThreads(config.NUM_HILOS_OPENCV)

def redimensionar_si_es_necesario(imagen_bgr, max_dim=config.MAX_DIM_IMAGEN):
    """
    Redimensiona la imagen si supera max_dim px.
    """
    alto, ancho = imagen_bgr.shape[:2]
    dim_mayor = max(alto, ancho)
    
    if dim_mayor > max_dim:
        escala = max_dim / float(dim_mayor)
        nuevo_ancho = int(ancho * escala)
        nuevo_alto = int(alto * escala)
        return cv2.resize(imagen_bgr, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    
    return imagen_bgr

def cargar_imagen_corregida_exif(origen_imagen):
    """
    Carga una imagen respetando la orientación EXIF de la cámara del celular.
    """
    if isinstance(origen_imagen, (bytes, bytearray)):
        img_pil = Image.open(io.BytesIO(origen_imagen))
    else:
        img_pil = Image.open(origen_imagen)
    
    img_pil = ImageOps.exif_transpose(img_pil)
    img_pil = img_pil.convert('RGB')
    imagen_np = np.array(img_pil)
    
    return cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

def ordenar_puntos_esquinas(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def detectar_y_recortar_documento(imagen_bgr, max_dim_analisis=config.MAX_DIM_MINIATURA_ANALISIS):
    """
    Detección ultra-conservadora de bordes externos de la hoja de papel.
    """
    alto_orig, ancho_orig = imagen_bgr.shape[:2]
    
    escala = max_dim_analisis / float(max(alto_orig, ancho_orig))
    if escala < 1.0:
        ancho_mini = int(ancho_orig * escala)
        alto_mini = int(alto_orig * escala)
        mini = cv2.resize(imagen_bgr, (ancho_mini, alto_mini), interpolation=cv2.INTER_AREA)
    else:
        mini = imagen_bgr.copy()
        escala = 1.0

    grises = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grises, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(canny, kernel, iterations=1)

    contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]

    area_total_mini = mini.shape[0] * mini.shape[1]
    esquinas_halladas = None

    for c in contornos:
        area_c = cv2.contourArea(c)
        if area_c < area_total_mini * config.PORCENTAJE_MIN_COBERTURA_PAPEL:
            continue

        perimetro = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            x, y, w, h = cv2.boundingRect(pts)
            aspect_ratio = w / float(h)
            if 0.4 <= aspect_ratio <= 2.2:
                esquinas_halladas = pts
                break

    if esquinas_halladas is None:
        return imagen_bgr, False

    pts_mini = ordenar_puntos_esquinas(esquinas_halladas.astype("float32"))
    pts_orig = pts_mini / escala

    (tl, tr, br, bl) = pts_orig

    ancho_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_ancho = max(int(ancho_a), int(ancho_b))

    alto_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_alto = max(int(alto_a), int(alto_b))

    dst = np.array([
        [0, 0],
        [max_ancho - 1, 0],
        [max_ancho - 1, max_alto - 1],
        [0, max_alto - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_orig, dst)
    recortada = cv2.warpPerspective(imagen_bgr, M, (max_ancho, max_alto))

    return recortada, True

def corregir_iluminacion_suave(imagen_grises):
    alto, ancho = imagen_grises.shape[:2]
    sigma_val = max(35, int(min(alto, ancho) * 0.05))
    
    fondo = cv2.GaussianBlur(imagen_grises, (0, 0), sigmaX=sigma_val, sigmaY=sigma_val)
    
    imagen_float = imagen_grises.astype(np.float32)
    fondo_float = fondo.astype(np.float32)
    
    resultado = (imagen_float / (fondo_float + 1e-5)) * 255.0
    return np.clip(resultado, 0, 255).astype(np.uint8)

def blanquear_fondo_y_resaltar(imagen_grises):
    iluminada = corregir_iluminacion_suave(imagen_grises)
    
    clahe = cv2.createCLAHE(clipLimit=config.CLIP_LIMIT_CLAHE, tileGridSize=(8, 8))
    clahe_img = clahe.apply(iluminada)
    
    val_min, val_max = np.percentile(clahe_img, (0.5, 98.0))
    estirada = np.clip((clahe_img - val_min) * (255.0 / (val_max - val_min + 1e-5)), 0, 255).astype(np.uint8)
    
    table = np.array([min(255, int((i / 255.0) ** 0.85 * 265)) for i in range(256)]).astype("uint8")
    resultado = cv2.LUT(estirada, table)
    
    desenfocada = cv2.GaussianBlur(resultado, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.addWeighted(resultado, config.FUERZA_NITIDEZ, desenfocada, -(config.FUERZA_NITIDEZ - 1.0), 0)

def binarizar_sauvola(imagen_grises, window_size=21, c_val=10):
    if window_size % 2 == 0:
        window_size += 1
    return cv2.adaptiveThreshold(
        imagen_grises, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, window_size, C=c_val
    )

def procesar_imagen_a_bytes(origen_imagen, modo=config.MODO_PROCESAMIENTO_DEFECTO, auto_crop=config.USAR_AUTO_CROP_DEFECTO):
    """
    Procesa una sola foto y devuelve los bytes JPEG comprimidos en memoria RAM.
    Libera inmediatamente las matrices pesadas de OpenCV para mantener el uso de RAM al mínimo.
    """
    img_bgr = cargar_imagen_corregida_exif(origen_imagen)

    if auto_crop:
        img_bgr, _ = detectar_y_recortar_documento(img_bgr)

    img_bgr = redimensionar_si_es_necesario(img_bgr, max_dim=config.MAX_DIM_IMAGEN)
    grises = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if modo == "magico":
        resultado_final = blanquear_fondo_y_resaltar(grises)
    elif modo == "otsu":
        ilum = corregir_iluminacion_suave(grises)
        resultado_final = binarizar_sauvola(ilum)
    else:
        resultado_final = grises

    exito, buffer_jpg = cv2.imencode(".jpg", resultado_final, [cv2.IMWRITE_JPEG_QUALITY, config.CALIDAD_JPEG])
    if not exito:
        raise RuntimeError("Error al codificar imagen en memoria RAM")

    bytes_resultado = buffer_jpg.tobytes()

    # LIBERACIÓN DE MEMORIA RAM EXPLÍCITA POR PÁGINA
    del img_bgr, grises, resultado_final, buffer_jpg
    if config.LIMPIAR_RAM_POR_PAGINA:
        gc.collect()

    return bytes_resultado

def procesar_lote_documentos(lista_origenes_imagenes, ruta_pdf_salida, modo=config.MODO_PROCESAMIENTO_DEFECTO, auto_crop=config.USAR_AUTO_CROP_DEFECTO):
    """
    Procesa un lote de N imágenes (ej. 30 o 100 fotos) página por página,
    empaquetándolas en un solo PDF multipágina sin acumular imágenes pesadas en RAM.
    """
    print(f"--> Procesando lote de {len(lista_origenes_imagenes)} imágenes a PDF: {ruta_pdf_salida}")
    os.makedirs(os.path.dirname(ruta_pdf_salida), exist_ok=True)

    buffers_jpeg = []
    for i, origen in enumerate(lista_origenes_imagenes, 1):
        bytes_jpg = procesar_imagen_a_bytes(origen, modo=modo, auto_crop=auto_crop)
        buffers_jpeg.append(bytes_jpg)

    with open(ruta_pdf_salida, "wb") as f:
        f.write(img2pdf.convert(buffers_jpeg))

    print(f"--> PDF multipágina generado con éxito en: {ruta_pdf_salida}\n")

def procesar_documento(origen_imagen_entrada, ruta_pdf_salida, modo=config.MODO_PROCESAMIENTO_DEFECTO, auto_crop=config.USAR_AUTO_CROP_DEFECTO):
    """
    Procesa una sola imagen a PDF (Wrapper para compatibilidad).
    """
    bytes_jpg = procesar_imagen_a_bytes(origen_imagen_entrada, modo=modo, auto_crop=auto_crop)
    os.makedirs(os.path.dirname(ruta_pdf_salida), exist_ok=True)
    with open(ruta_pdf_salida, "wb") as f:
        f.write(img2pdf.convert(bytes_jpg))
    print(f"--> PDF generado con éxito en: {ruta_pdf_salida}\n")

if __name__ == "__main__":
    carpeta_input = "input"
    carpeta_output = "output"

    if not os.path.exists(carpeta_input):
        os.makedirs(carpeta_input)

    if not os.path.exists(carpeta_output):
        os.makedirs(carpeta_output)

    archivos_input = [os.path.join(carpeta_input, f) for f in os.listdir(carpeta_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not archivos_input:
        print(f"Coloca fotos de prueba (.jpg o .png) en la carpeta '{carpeta_input}'.")
    else:
        # Generar PDF multipágina con todas las fotos encontradas
        ruta_pdf_lote = os.path.join(carpeta_output, "documento_procesado_lote.pdf")
        procesar_lote_documentos(archivos_input, ruta_pdf_lote)
