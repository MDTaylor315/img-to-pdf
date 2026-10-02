import os
import io
import gc
import cv2
import numpy as np
import img2pdf
from PIL import Image, ImageOps

import importlib

# Detector de esquinas via Deep Learning (ONNX) — Tier 0
try:
    from . import doc_detector as _doc_detector
except (ImportError, ValueError):
    import doc_detector as _doc_detector

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
    Aplica transformacion de perspectiva para desdoblar la hoja y estirarla
    de esquina a esquina a un rectangulo perfecto, eliminando fondos diagonales.
    """
    rect = ordenar_cuatro_puntos(pts)
    (tl, tr, br, bl) = rect

    # Calcular ancho proyectado como promedio de los dos lados horizontales.
    # Usar el promedio en lugar del maximo da un aspect ratio mas estable cuando
    # las esquinas tienen pequenas imprecisiones (p.ej. detector DL con heatmap 128px).
    ancho_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    ancho_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_ancho = int((ancho_a + ancho_b) / 2)

    # Calcular alto proyectado como promedio de los dos lados verticales
    alto_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    alto_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_alto = int((alto_a + alto_b) / 2)

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

def recortar_bordes_residuales(desdoblada_bgr, max_pct=0.035, umbral_grad=15.0):
    """
    Detecta y poda finas franjas residuales de fondo oscuro (mesa, escritorio)
    en el perímetro del documento desdoblado (máximo max_pct, ej. 3.5%).
    Solo recorta si el borde exterior es efectivamente fondo no-papel (< 130).
    """
    h, w = desdoblada_bgr.shape[:2]
    gray = cv2.cvtColor(desdoblada_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Borde inferior
    max_trim_b = int(h * max_pct)
    cut_b = 0
    if h > 100 and max_trim_b > 2:
        outer_b = gray[h-3:h, int(0.10 * w):int(0.90 * w)].mean()
        if outer_b < 130:
            dy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            bot_strip = dy[h - max_trim_b:h, int(0.10 * w):int(0.90 * w)].mean(axis=1)
            peaks = np.where(bot_strip > umbral_grad)[0]
            if len(peaks) > 0:
                cut_b = max_trim_b - peaks[0] + 1

    # 2. Borde superior
    max_trim_t = int(h * max_pct)
    cut_t = 0
    if h > 100 and max_trim_t > 2:
        outer_t = gray[:3, int(0.10 * w):int(0.90 * w)].mean()
        if outer_t < 130:
            dy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            top_strip = dy[:max_trim_t, int(0.10 * w):int(0.90 * w)].mean(axis=1)
            peaks = np.where(top_strip > umbral_grad)[0]
            if len(peaks) > 0:
                cut_t = peaks[-1] + 1

    # 3. Borde izquierdo
    max_trim_l = int(w * max_pct)
    cut_l = 0
    if w > 100 and max_trim_l > 2:
        outer_l = gray[int(0.10 * h):int(0.90 * h), :3].mean()
        if outer_l < 130:
            dx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            left_strip = dx[int(0.10 * h):int(0.90 * h), :max_trim_l].mean(axis=0)
            peaks = np.where(left_strip > umbral_grad)[0]
            if len(peaks) > 0:
                cut_l = peaks[-1] + 1

    # 4. Borde derecho
    max_trim_r = int(w * max_pct)
    cut_r = 0
    if w > 100 and max_trim_r > 2:
        outer_r = gray[int(0.10 * h):int(0.90 * h), w-3:w].mean()
        if outer_r < 130:
            dx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            right_strip = dx[int(0.10 * h):int(0.90 * h), w - max_trim_r:w].mean(axis=0)
            peaks = np.where(right_strip > umbral_grad)[0]
            if len(peaks) > 0:
                cut_r = max_trim_r - peaks[0] + 1

    y1 = cut_t
    y2 = h - cut_b if cut_b > 0 else h
    x1 = cut_l
    x2 = w - cut_r if cut_r > 0 else w

    if (x2 - x1) > 100 and (y2 - y1) > 100:
        return desdoblada_bgr[y1:y2, x1:x2]
    return desdoblada_bgr


def detectar_y_recortar_documento(imagen_bgr, max_dim_analisis=config.MAX_DIM_MINIATURA_ANALISIS):
    """
    Detección de bordes y perspectiva de la hoja de papel con control inteligente.
    Prioridad:
    0. Deep Learning (ONNX): YOLO localiza el documento + LCNet regresa las 4 esquinas exactas.
       Se aplica validación de cobertura frente a la silueta de papel para no amputar hojas dobladas/arrugadas.
    1. Canny + RETR_LIST para aislar el papel blanco de laptops o fondos oscuros/claros.
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

    # Estimación rápida de cobertura de papel global
    _, thresh_pre = cv2.threshold(grises, 125, 255, cv2.THRESH_BINARY)
    k_pre = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    clean_pre = cv2.morphologyEx(thresh_pre, cv2.MORPH_CLOSE, k_pre)
    cnts_pre, _ = cv2.findContours(clean_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_max_pre = cv2.contourArea(max(cnts_pre, key=cv2.contourArea)) if cnts_pre else 0
    pct_papel_global = area_max_pre / float(area_total_mini)

    # --- TIER 0: DETECCIÓN DE 4 ESQUINAS VIA DEEP LEARNING (ONNX) ---
    if getattr(config, "USAR_DETECTOR_DL", True):
        esquinas_dl = _doc_detector.detectar_esquinas_documento(imagen_bgr)
        if esquinas_dl is not None:
            area_dl = cv2.contourArea(esquinas_dl) / float(ancho_orig * alto_orig)
            # Guardia de Cobertura: si OpenCV ve que la hoja cubre >= 55% de la foto pero DL
            # sólo detectó una fracción (< 55% o < 65% de la silueta real, ej. manual arrugado en Pag 1 y Pag 3),
            # descartamos DL para no cortar la página a la mitad y dejamos que OpenCV capture el documento completo.
            usar_dl = True
            if pct_papel_global >= 0.55 and area_dl < 0.50:
                usar_dl = False
            elif pct_papel_global >= 0.65 and area_dl < (pct_papel_global * 0.60):
                usar_dl = False

            if usar_dl:
                desdoblada, ok = aplicar_perspectiva_cuatro_puntos(imagen_bgr, esquinas_dl)
                if ok:
                    desdoblada = recortar_bordes_residuales(desdoblada)
                    return desdoblada, True

    # --- TIER 1: DETECCION DE 4 ESQUINAS DE PAPEL Y CORRECCION DE PERSPECTIVA ---
    blur = cv2.GaussianBlur(grises, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 150)
    k_canny = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(canny, k_canny, iterations=2)

    cnts_list, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidatos_quad = []

    for c in sorted(cnts_list, key=cv2.contourArea, reverse=True)[:8]:
        area_c = cv2.contourArea(c)
        pct = area_c / float(area_total_mini)
        if 0.20 <= pct <= 0.96:
            # Si la foto es un plano cerrado donde el papel llena casi todo (pct_papel_global > 0.88),
            # no aceptar cuadriláteros que cubran menos del 80% (son tablas internas, ej. Pag 2)
            if pct_papel_global > 0.88 and pct < 0.80:
                continue

            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                mask = np.zeros_like(grises)
                cv2.drawContours(mask, [approx], -1, 255, -1)
                brillo_promedio = cv2.mean(grises, mask=mask)[0]

                mask_ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))) - mask
                brillo_ring = cv2.mean(grises, mask=mask_ring)[0]

                if brillo_ring > 200:
                    continue
                if (brillo_promedio - brillo_ring >= 15) or (brillo_ring < 145):
                    candidatos_quad.append((brillo_promedio, pct, approx))

    if candidatos_quad:
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

            # Refinamiento contra esquinas con escritorio visible o papel secundario desfasado:
            # Si la fila o columna exterior contiene fondo/escritorio (0 en clean_paper),
            # avanzamos el límite hacia el interior hasta delimitar exclusivamente la hoja principal.
            max_dy = int((y2 - y1) * 0.25)
            max_dx = int((x2 - x1) * 0.25)
            y1_lim = y1 + max_dy
            while y1 < y1_lim and np.mean(clean_paper[y1, x1:x2] == 0) > 0.15:
                y1 += 1
            y2_lim = y2 - max_dy
            while y2 > y2_lim and np.mean(clean_paper[y2 - 1, x1:x2] == 0) > 0.15:
                y2 -= 1
            x1_lim = x1 + max_dx
            while x1 < x1_lim and np.mean(clean_paper[y1:y2, x1] == 0) > 0.15:
                x1 += 1
            x2_lim = x2 - max_dx
            while x2 > x2_lim and np.mean(clean_paper[y1:y2, x2 - 1] == 0) > 0.15:
                x2 -= 1

        rx1 = int(x1 / escala)
        ry1 = int(y1 / escala)
        rx2 = int(x2 / escala)
        ry2 = int(y2 / escala)

        # Margen de seguridad mínimo (~0.4%) para podar rasgados de esquinas sin amputar sellos
        if pct_area < 0.94:
            pad_x = int(0.004 * (rx2 - rx1))
            pad_y = int(0.003 * (ry2 - ry1))
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

    # Evaluar los márgenes exteriores: si son >95% papel, puede ser una foto cerrada del recuadro móvil
    top_p = np.mean(clean_rot[0, :] > 0)
    bot_p = np.mean(clean_rot[-1, :] > 0)
    left_p = np.mean(clean_rot[:, 0] > 0)
    right_p = np.mean(clean_rot[:, -1] > 0)
    if top_p > 0.95 and bot_p > 0.95 and left_p > 0.95 and right_p > 0.95 and pct_area > 0.85:
        trim_y1 = 0
        trim_y2 = alto_mini
        trim_x1 = 0
        trim_x2 = ancho_mini
        max_trim_y = int(alto_mini * 0.10)
        max_trim_x = int(ancho_mini * 0.10)

        for y in range(0, max_trim_y):
            if np.mean(clean_rot[y, :] > 0) >= 0.97:
                trim_y1 = y
                break
        for y in range(alto_mini - 1, alto_mini - 1 - max_trim_y, -1):
            if np.mean(clean_rot[y, :] > 0) >= 0.97:
                trim_y2 = y + 1
                break
        for x in range(0, max_trim_x):
            if np.mean(clean_rot[:, x] > 0) >= 0.97:
                trim_x1 = x
                break
        for x in range(ancho_mini - 1, ancho_mini - 1 - max_trim_x, -1):
            if np.mean(clean_rot[:, x] > 0) >= 0.97:
                trim_x2 = x + 1
                break

        recorto = (trim_y1 > 1 or trim_y2 < alto_mini - 1 or
                   trim_x1 > 1 or trim_x2 < ancho_mini - 1)
        if recorto:
            rx1 = max(0, int(trim_x1 / escala))
            ry1 = max(0, int(trim_y1 / escala))
            rx2 = min(ancho_orig, int(trim_x2 / escala))
            ry2 = min(alto_orig, int(trim_y2 / escala))
            if rx2 - rx1 > 100 and ry2 - ry1 > 100:
                return imagen_rotada[ry1:ry2, rx1:rx2], True
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
    angulo_onnx = clasificar_angulo_orientacion_onnx(imagen_bgr)
    if angulo_onnx is not None:
        if angulo_onnx == 90:
            return cv2.rotate(imagen_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), True
        elif angulo_onnx == 180:
            return cv2.rotate(imagen_bgr, cv2.ROTATE_180), True
        elif angulo_onnx == 270:
            return cv2.rotate(imagen_bgr, cv2.ROTATE_90_CLOCKWISE), True
        return imagen_bgr, False

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


def corregir_inclinacion_fina_texto(imagen_bgr, max_angulo=1.5):
    """
    Detecta y corrige cualquier inclinación residual leve (0.25° a 1.5°)
    en las líneas de texto del documento (subpixel deskew).
    Realiza recorte inscrito automático para eliminar cualquier triángulo blanco en bordes.
    """
    h, w = imagen_bgr.shape[:2]
    escala = 1000.0 / float(max(h, w))
    mini = cv2.resize(imagen_bgr, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(30 * escala) | 1, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        dilated, 1, np.pi / 1800, threshold=80,
        minLineLength=int(100 * escala), maxLineGap=int(15 * escala)
    )
    if lines is None or len(lines) < 10:
        return imagen_bgr, False

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l.flatten()
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > int(80 * escala):
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) <= max_angulo:
                angles.append(angle)

    if len(angles) < 10:
        return imagen_bgr, False

    median_angle = float(np.median(angles))

    if abs(median_angle) >= 0.25:
        centro = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(centro, median_angle, 1.0)
        rotada = cv2.warpAffine(
            imagen_bgr, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        # Recorte inscrito para garantizar cero fondo vacío
        rad = np.radians(abs(median_angle))
        sin_a = np.sin(rad)
        pad_w = int(np.ceil(0.5 * h * sin_a))
        pad_h = int(np.ceil(0.5 * w * sin_a))
        if pad_w > 0 or pad_h > 0:
            rotada = rotada[pad_h:h-pad_h, pad_w:w-pad_w]
        return rotada, True

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
        # Enderezado fino de líneas de texto (subpixel deskew: 0.25° a 7.0°)
        img_bgr, _ = corregir_inclinacion_fina_texto(img_bgr)

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
