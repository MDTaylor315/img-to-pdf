import os
import io
import gc
import cv2
import numpy as np
import img2pdf
from PIL import Image, ImageOps

import importlib

# Intentar cargar pytesseract opcionalmente vía importlib para evitar advertencias de linter estático
try:
    pytesseract = importlib.import_module("pytesseract")
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False


# Cargar constantes de configuración centralizadas
try:
    from . import config
except (ImportError, ValueError):
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
    origen = io.BytesIO(origen_imagen) if isinstance(origen_imagen, (bytes, bytearray)) else origen_imagen

    with Image.open(origen) as img_pil:
        if img_pil.format not in config.FORMATOS_IMAGEN_PERMITIDOS:
            raise ValueError(
                "Formato de imagen no permitido. Use JPEG, PNG o WebP."
            )

        ancho, alto = img_pil.size
        if ancho <= 0 or alto <= 0:
            raise ValueError("La imagen no tiene dimensiones válidas.")
        if ancho * alto > config.MAX_PIXELES_POR_FOTO:
            raise ValueError(
                "La imagen supera el máximo de %s megapíxeles."
                % (config.MAX_PIXELES_POR_FOTO // 1_000_000)
            )

        img_pil = ImageOps.exif_transpose(img_pil)
        img_pil = img_pil.convert('RGB')
        imagen_np = np.asarray(img_pil)
    
    return cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

def recortar_recortes_secundarios_papel(mini_thresh, x, y, w, h):
    """
    Filtra recortes o pedazos de papel secundarios pegados al costado de la hoja principal (ej. Hoja 5).
    """
    crop_mask = mini_thresh[y:y+h, x:x+w]
    col_sums = np.sum(crop_mask > 0, axis=0)
    max_col = np.max(col_sums)
    if max_col == 0:
        return x, y, w, h
    valid_cols = np.where(col_sums >= 0.45 * max_col)[0]
    if len(valid_cols) > 0:
        start_col = valid_cols[0]
        end_col = valid_cols[-1]
        new_x = x + start_col
        new_w = end_col - start_col + 1
        return new_x, y, new_w, h
    return x, y, w, h

def ordenar_cuatro_puntos(pts):
    """
    Ordena 4 puntos en el orden: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left (suma x+y mínima)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (suma x+y máxima)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right (diferencia y-x mínima)
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left (diferencia y-x máxima)
    return rect

def aplicar_perspectiva_cuatro_puntos(imagen_bgr, pts):
    """
    Aplica transformación de perspectiva para desdoblar la hoja y estirarla
    de esquina a esquina a un rectángulo perfecto, eliminando fondos diagonales.
    """
    rect = ordenar_cuatro_puntos(pts)
    (tl, tr, br, bl) = rect

    # Calcular ancho proyectado
    ancho_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_ancho = max(int(ancho_a), int(ancho_b))

    # Calcular alto proyectado
    alto_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_alto = max(int(alto_a), int(alto_b))

    if max_ancho < 50 or max_alto < 50:
        return imagen_bgr, False

    destino = np.array([
        [0, 0],
        [max_ancho - 1, 0],
        [max_ancho - 1, max_alto - 1],
        [0, max_alto - 1]
    ], dtype=np.float32)

    matriz = cv2.getPerspectiveTransform(rect, destino)
    desdoblada = cv2.warpPerspective(imagen_bgr, matriz, (max_ancho, max_alto), flags=cv2.INTER_LINEAR)
    return desdoblada, True

def detectar_y_recortar_documento(imagen_bgr, max_dim_analisis=config.MAX_DIM_MINIATURA_ANALISIS):
    """
    Detección de bordes y perspectiva de la hoja de papel.
    Prioridad: 
    1. Si detecta las 4 esquinas de la hoja (incluso en ángulo o sobre laptops),
       aplica transformación de perspectiva para que las esquinas del archivo coincidan
       estrictamente con las esquinas físicas de la hoja (sin teclado ni fondo).
    2. Respaldo conservador si la toma está en plano cerrado o tiene adjuntos.
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

    ancho_mini, alto_mini = mini.shape[1], mini.shape[0]
    area_total_mini = ancho_mini * alto_mini
    grises = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)

    # --- TIER 1: DETECCIÓN DE 4 ESQUINAS DE PAPEL Y CORRECCIÓN DE PERSPECTIVA ---
    # Usar Canny + RETR_LIST para aislar el papel blanco de laptops o fondos oscuros/claros
    blur = cv2.GaussianBlur(grises, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 150)
    k_canny = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(canny, k_canny, iterations=2)

    cnts_list, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos_quad = []

    # Evaluar solo los contornos principales ordenados por área para máxima velocidad
    for c in sorted(cnts_list, key=cv2.contourArea, reverse=True)[:8]:
        area_c = cv2.contourArea(c)
        pct = area_c / float(area_total_mini)
        if 0.20 <= pct <= 0.88:
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                mask = np.zeros_like(grises)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                brillo_promedio = cv2.mean(grises, mask=mask)[0]

                # Comprobar el anillo exterior al cuadrilátero para distinguir
                # el borde real de la hoja vs una tabla interna impresa en papel blanco
                mask_ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))) - mask
                brillo_ring = cv2.mean(grises, mask=mask_ring)[0]

                # La hoja de papel sobre una mesa o teclado tiene un exterior oscuro (brillo_ring < 125)
                # o una caída notable de luminosidad. Una tabla interna tiene más papel blanco afuera (~180).
                if (brillo_promedio - brillo_ring >= 25) or (brillo_ring < 125):
                    candidatos_quad.append((brillo_promedio, pct, approx))

    if candidatos_quad:
        # La hoja de papel es siempre el cuadrilátero con mayor reflectancia/brillo blanco
        candidatos_quad.sort(key=lambda item: item[0], reverse=True)
        mejor_quad = candidatos_quad[0]
        pts_orig = (mejor_quad[2].reshape(-1, 2) / escala).astype(np.float32)
        desdoblada, ok = aplicar_perspectiva_cuatro_puntos(imagen_bgr, pts_orig)
        if ok:
            return desdoblada, True

    # --- TIER 2: DELIMITACIÓN DE HOJA Y ELIMINACIÓN DE FONDO/ESCRITORIO ---
    _, thresh_paper = cv2.threshold(grises, 125, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    clean_paper = cv2.morphologyEx(thresh_paper, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(clean_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return imagen_bgr, False

    c = max(contornos, key=cv2.contourArea)
    area_c = cv2.contourArea(c)
    pct_area = area_c / float(area_total_mini)

    # Si el contorno es diminuto (< 20%), no recortar
    if pct_area < config.PORCENTAJE_MIN_COBERTURA_PAPEL:
        return imagen_bgr, False

    # 1. Intentar recorte por rectángulo inscrito de cuadrilátero envolvente (Garantía de Cero Escritorio)
    hull = cv2.convexHull(c)
    quad_pts = None
    for eps in [0.015, 0.02, 0.025, 0.03, 0.04]:
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            pts = approx.reshape(-1, 2)
            rect = np.zeros((4, 2), dtype='float32')
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # TL
            rect[2] = pts[np.argmax(s)]  # BR
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # TR
            rect[3] = pts[np.argmax(diff)] # BL
            quad_pts = rect
            break

    if quad_pts is not None:
        TL, TR, BR, BL = quad_pts
        # Si hay un papel secundario o recibo adjunto en la esquina inferior derecha
        br_x = TR[0] if BR[0] > TR[0] + (50 * escala) else BR[0]
        br_y = BL[1] if BR[0] > TR[0] + (50 * escala) else BR[1]

        if pct_area > 0.94:
            # En fotos cerradas donde la hoja llena casi toda la pantalla (ej. Hoja 2),
            # no recortar esquinas exteriores ni aplicar paddings para no amputar notas al margen ("OBSERVACION")
            x1 = int(max(0, min(TL[0], BL[0])))
            y1 = int(max(0, min(TL[1], TR[1])))
            x2 = int(min(ancho_mini, max(TR[0], br_x)))
            y2 = int(min(alto_mini, max(BL[1], br_y)))
        else:
            x1 = int(max(0, max(TL[0], BL[0])))
            y1 = int(max(0, max(TL[1], TR[1])))
            x2 = int(min(ancho_mini, min(TR[0], br_x)))
            y2 = int(min(alto_mini, min(BL[1], br_y)))

            # Refinamiento contra esquinas con escritorio visible o papel secundario desfasado (ej. gd-1, gd-2):
            # Si la fila o columna exterior contiene fondo/escritorio (0 en clean_paper),
            # avanzamos el límite hacia el interior hasta delimitar exclusivamente la hoja principal.
            max_dy = int((y2 - y1) * 0.35)
            max_dx = int((x2 - x1) * 0.35)
            y1_lim = y1 + max_dy
            while y1 < y1_lim and np.mean(clean_paper[y1, x1:x2] == 0) > 0.03:
                y1 += 1
            y2_lim = y2 - max_dy
            while y2 > y2_lim and np.mean(clean_paper[y2 - 1, x1:x2] == 0) > 0.03:
                y2 -= 1
            x1_lim = x1 + max_dx
            while x1 < x1_lim and np.mean(clean_paper[y1:y2, x1] == 0) > 0.03:
                x1 += 1
            x2_lim = x2 - max_dx
            while x2 > x2_lim and np.mean(clean_paper[y1:y2, x2 - 1] == 0) > 0.03:
                x2 -= 1

        rx1 = int(x1 / escala)
        ry1 = int(y1 / escala)
        rx2 = int(x2 / escala)
        ry2 = int(y2 / escala)

        # Margen de seguridad fino (~1%) para podar rasgados de esquinas o sombras perimetrales
        if pct_area < 0.94:
            pad_x = int(0.012 * (rx2 - rx1))
            pad_y = int(0.008 * (ry2 - ry1))
            rx1 = min(ancho_orig, rx1 + pad_x)
            rx2 = max(0, rx2 - pad_x)
            ry1 = min(alto_orig, ry1 + pad_y)
            ry2 = max(0, ry2 - pad_y)

        if rx2 - rx1 > 100 and ry2 - ry1 > 100:
            return imagen_bgr[ry1:ry2, rx1:rx2], True

    # 2. Respaldo: Corrección de leve inclinación (deskew) y delimitación ortogonal
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90

    if abs(angle) > 0.5:
        centro = (ancho_orig / 2.0, alto_orig / 2.0)
        M_orig = cv2.getRotationMatrix2D(centro, angle, 1.0)
        imagen_rotada = cv2.warpAffine(imagen_bgr, M_orig, (ancho_orig, alto_orig), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        mini_centro = (ancho_mini / 2.0, alto_mini / 2.0)
        M_mini = cv2.getRotationMatrix2D(mini_centro, angle, 1.0)
        clean_rot = cv2.warpAffine(clean_paper, M_mini, (ancho_mini, alto_mini), flags=cv2.INTER_NEAREST)
    else:
        imagen_rotada = imagen_bgr
        clean_rot = clean_paper

    rot_cnts, _ = cv2.findContours(clean_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not rot_cnts:
        return imagen_bgr, False
    c_rot = max(rot_cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c_rot)

    # Si la imagen ya es 100% papel blanco en todos sus márgenes exteriores, no recortar
    top_p = np.mean(clean_rot[0, :] > 0)
    bot_p = np.mean(clean_rot[-1, :] > 0)
    left_p = np.mean(clean_rot[:, 0] > 0)
    right_p = np.mean(clean_rot[:, -1] > 0)
    if top_p > 0.95 and bot_p > 0.95 and left_p > 0.95 and right_p > 0.95 and pct_area > 0.85:
        return imagen_bgr, False

    # Delimitar límites donde el cuerpo del papel es continuo (sin escritorio)
    mid_y1 = by + int(bh * 0.25)
    mid_y2 = by + int(bh * 0.75)
    cols_in_body = np.mean(clean_rot[mid_y1:mid_y2, :] > 0, axis=0)
    valid_cols = np.where(cols_in_body > 0.35)[0]
    if len(valid_cols) == 0:
        return imagen_bgr, False
    x_left = valid_cols[0]
    x_right = valid_cols[-1]

    y_top = by
    for y in range(by, by + int(bh * 0.12)):
        if np.mean(clean_rot[y, x_left:x_right] > 0) > 0.35:
            y_top = y
            break

    y_bot = by + bh
    for y in range(by + bh - 1, by + int(bh * 0.88), -1):
        if np.mean(clean_rot[y, x_left:x_right] > 0) > 0.85:
            y_bot = y
            break

    for x in range(x_left, x_left + int((x_right - x_left) * 0.12)):
        if np.mean(clean_rot[y_top:y_bot, x] > 0) > 0.50:
            x_left = x
            break

    for x in range(x_right, x_right - int((x_right - x_left) * 0.12), -1):
        if np.mean(clean_rot[y_top:y_bot, x] > 0) > 0.50:
            x_right = x
            break

    rx1 = max(0, int(x_left / escala))
    ry1 = max(0, int(y_top / escala))
    rx2 = min(ancho_orig, int(x_right / escala))
    ry2 = min(alto_orig, int(y_bot / escala))

    if rx2 - rx1 < 50 or ry2 - ry1 < 50:
        return imagen_bgr, False

    recortada = imagen_rotada[ry1:ry2, rx1:rx2]
    return recortada, True

    return imagen_bgr, False

_RED_ORIENTACION_ONNX = None
_ERROR_CARGA_MODELO_ONNX = False

def obtener_red_orientacion():
    """
    Carga el modelo ONNX de orientación en memoria una sola vez (Lazy Singleton).
    """
    global _RED_ORIENTACION_ONNX, _ERROR_CARGA_MODELO_ONNX
    if _RED_ORIENTACION_ONNX is not None:
        return _RED_ORIENTACION_ONNX
    if _ERROR_CARGA_MODELO_ONNX:
        return None

    if getattr(config, "USAR_MODELO_ORIENTACION_ONNX", False):
        model_path = getattr(config, "MODELO_ORIENTACION_PATH", None)
        if model_path and os.path.isfile(model_path):
            try:
                _RED_ORIENTACION_ONNX = cv2.dnn.readNetFromONNX(model_path)
                return _RED_ORIENTACION_ONNX
            except Exception as e:
                print(f"[MontanoImagen] Advertencia: No se pudo cargar el modelo ONNX: {e}")
                _ERROR_CARGA_MODELO_ONNX = True
        else:
            _ERROR_CARGA_MODELO_ONNX = True
    return None

def clasificar_angulo_orientacion_onnx(imagen_bgr):
    """
    Evalúa la imagen con la red ONNX (ejecutada en cv2.dnn) y devuelve el ángulo: 0, 90, 180 o 270.
    Inferencia promedio: ~7 ms en CPU.
    """
    net = obtener_red_orientacion()
    if net is None:
        return None

    try:
        h, w = imagen_bgr.shape[:2]
        scale = 256.0 / float(min(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(imagen_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        start_x = (new_w - 224) // 2
        start_y = (new_h - 224) // 2
        cropped = resized[start_y:start_y + 224, start_x:start_x + 224]

        blob_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (blob_rgb - mean) / std
        chw = np.transpose(norm, (2, 0, 1))[None, ...]

        net.setInput(chw)
        pred_output = net.forward()[0]
        etiquetas = [0, 90, 180, 270]
        idx = int(np.argmax(pred_output))
        return etiquetas[idx]
    except Exception as e:
        print(f"[MontanoImagen] Advertencia al inferir orientación ONNX: {e}")
        return None

def detectar_y_corregir_orientacion_texto(imagen_bgr, max_dim_analisis=config.MAX_DIM_MINIATURA_ANALISIS):
    """
    Detecta y corrige la orientación del documento (0°, 90°, 180°, 270°)
    para dejar el texto derecho y alineado a la lectura natural de la página.
    """
    # 1. Prioridad: Clasificador ONNX ultraligero (~7ms con cv2.dnn)
    angulo_onnx = clasificar_angulo_orientacion_onnx(imagen_bgr)
    if angulo_onnx is not None:
        if angulo_onnx == 90:
            return cv2.rotate(imagen_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), True
        elif angulo_onnx == 180:
            return cv2.rotate(imagen_bgr, cv2.ROTATE_180), True
        elif angulo_onnx == 270:
            return cv2.rotate(imagen_bgr, cv2.ROTATE_90_CLOCKWISE), True
        return imagen_bgr, False

    # 2. Respaldo secundario: Tesseract OSD si está disponible
    if HAS_PYTESSERACT and pytesseract is not None:
        try:
            img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
            osd = pytesseract.image_to_osd(Image.fromarray(img_rgb), output_type=pytesseract.Output.DICT)
            rot = osd.get('rotate', 0)
            if rot == 90:
                return cv2.rotate(imagen_bgr, cv2.ROTATE_90_CLOCKWISE), True
            elif rot == 180:
                return cv2.rotate(imagen_bgr, cv2.ROTATE_180), True
            elif rot == 270:
                return cv2.rotate(imagen_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), True
            return imagen_bgr, False
        except Exception:
            pass

    # 3. Respaldo terciario: Heurística geométrica de 90° (para fotos tomadas de lado)
    alto, ancho = imagen_bgr.shape[:2]
    max_dim = max(alto, ancho)

    if max_dim > max_dim_analisis:
        escala = float(max_dim_analisis) / max_dim
        mini_grises = cv2.cvtColor(
            cv2.resize(imagen_bgr, (int(ancho * escala), int(alto * escala)), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY
        )
    else:
        mini_grises = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        mini_grises, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    proj_h = np.var(np.sum(thresh, axis=1))
    proj_v = np.var(np.sum(thresh, axis=0))
    if ancho > alto and proj_v > 1.3 * proj_h:
        return cv2.rotate(imagen_bgr, cv2.ROTATE_90_CLOCKWISE), True

    return imagen_bgr, False



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

def procesar_imagen_a_bytes(
    origen_imagen,
    modo=config.MODO_PROCESAMIENTO_DEFECTO,
    auto_crop=config.USAR_AUTO_CROP_DEFECTO,
    auto_orientar=config.AUTO_ORIENTAR_TEXTO_DEFECTO,
):
    """
    Procesa una sola foto y devuelve los bytes JPEG comprimidos en memoria RAM.
    Libera inmediatamente las matrices pesadas de OpenCV para mantener el uso de RAM al mínimo.
    """
    img_bgr = cargar_imagen_corregida_exif(origen_imagen)

    # 1. AUTO-CROP PRIMERO (Encontrar y recortar la hoja según sus bordes naturales)
    if auto_crop:
        img_bgr, _ = detectar_y_recortar_documento(img_bgr)

    # 2. AUTO-ORIENTAR SEGUNDO (Ajustar si las líneas de texto están boca abajo o de lado)
    if auto_orientar:
        img_bgr, _ = detectar_y_corregir_orientacion_texto(img_bgr)

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


def convertir_imagenes_a_pdf_bytes(
    lista_origenes_imagenes,
    modo=config.MODO_PROCESAMIENTO_DEFECTO,
    auto_crop=config.USAR_AUTO_CROP_DEFECTO,
    auto_orientar=config.AUTO_ORIENTAR_TEXTO_DEFECTO,
):
    if not lista_origenes_imagenes:
        raise ValueError("Debe proporcionar al menos una imagen.")

    buffers_jpeg = [
        procesar_imagen_a_bytes(
            origen, modo=modo, auto_crop=auto_crop, auto_orientar=auto_orientar
        )
        for origen in lista_origenes_imagenes
    ]
    try:
        return img2pdf.convert(
            buffers_jpeg,
            rotation=img2pdf.Rotation.none,
        )
    finally:
        buffers_jpeg.clear()
        if config.LIMPIAR_RAM_POR_PAGINA:
            gc.collect()

def procesar_lote_documentos(
    lista_origenes_imagenes,
    ruta_pdf_salida,
    modo=config.MODO_PROCESAMIENTO_DEFECTO,
    auto_crop=config.USAR_AUTO_CROP_DEFECTO,
    auto_orientar=config.AUTO_ORIENTAR_TEXTO_DEFECTO,
):
    """
    Procesa un lote de N imágenes (ej. 30 o 100 fotos) página por página,
    empaquetándolas en un solo PDF multipágina sin acumular imágenes pesadas en RAM.
    """
    print(f"--> Procesando lote de {len(lista_origenes_imagenes)} imágenes a PDF: {ruta_pdf_salida}")
    os.makedirs(os.path.dirname(ruta_pdf_salida), exist_ok=True)

    with open(ruta_pdf_salida, "wb") as f:
        f.write(
            convertir_imagenes_a_pdf_bytes(
                lista_origenes_imagenes,
                modo=modo,
                auto_crop=auto_crop,
                auto_orientar=auto_orientar,
            )
        )

    print(f"--> PDF multipágina generado con éxito en: {ruta_pdf_salida}\n")

def procesar_documento(
    origen_imagen_entrada,
    ruta_pdf_salida,
    modo=config.MODO_PROCESAMIENTO_DEFECTO,
    auto_crop=config.USAR_AUTO_CROP_DEFECTO,
    auto_orientar=config.AUTO_ORIENTAR_TEXTO_DEFECTO,
):
    """
    Procesa una sola imagen a PDF (Wrapper para compatibilidad).
    """
    bytes_jpg = procesar_imagen_a_bytes(
        origen_imagen_entrada,
        modo=modo,
        auto_crop=auto_crop,
        auto_orientar=auto_orientar,
    )
    os.makedirs(os.path.dirname(ruta_pdf_salida), exist_ok=True)
    with open(ruta_pdf_salida, "wb") as f:
        f.write(img2pdf.convert(bytes_jpg, rotation=img2pdf.Rotation.none))
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
        procesar_lote_documentos(sorted(archivos_input), ruta_pdf_lote)
