"""
doc_detector.py — Detector de esquinas de documento con Deep Learning (ONNX)

Pipeline de 2 etapas:
  1. YOLO detector  -> localiza el bounding-box del documento en la imagen.
  2. LCNet regressor -> predice un heatmap de 4 esquinas dentro de ese recorte.

Ambos modelos pesan < 30 MB en total y corren en CPU pura (sin GPU) usando
ONNX Runtime, aniadiendo ~25-45 ms por foto frente al pipeline anterior.

Integracion: esta funcion expone una sola llamada publica:
    detectar_esquinas_documento(imagen_bgr) -> np.ndarray | None

Si devuelve None, el pipeline recae en la logica OpenCV de respaldo (Tiers 1 y 2).
"""

import os
import cv2
import numpy as np

from . import config

try:
    import onnxruntime as ort
    _ONNX_DISPONIBLE = True
except ImportError:
    _ONNX_DISPONIBLE = False

# ---------------------------------------------------------------------------
# Rutas de modelos desde config
# ---------------------------------------------------------------------------
_PATH_YOLO   = getattr(config, "MODELO_YOLO_DOC_PATH", os.path.join(os.path.dirname(__file__), "models", "yolo_doc_detector.onnx"))
_PATH_LCNET  = getattr(config, "MODELO_LCNET_DOC_PATH", os.path.join(os.path.dirname(__file__), "models", "lcnet_doc_corners.onnx"))

# ---------------------------------------------------------------------------
# Singletons — se cargan una sola vez en la primera llamada (Lazy Init)
# ---------------------------------------------------------------------------
_sess_yolo  = None
_sess_lcnet = None
_carga_fallida = False   # Si falla la carga, no reintentamos en cada peticion


def _cargar_modelos():
    """
    Carga ambas sesiones ONNX en memoria.
    Retorna True si ambos modelos se cargaron con exito, False si alguno falla.
    """
    global _sess_yolo, _sess_lcnet, _carga_fallida

    if _carga_fallida:
        return False
    if _sess_yolo is not None and _sess_lcnet is not None:
        return True

    if not _ONNX_DISPONIBLE:
        print("[DocDetector] onnxruntime no esta instalado. Recayendo en OpenCV heuristico.")
        _carga_fallida = True
        return False

    num_hilos = getattr(config, "NUM_HILOS_OPENCV", 1)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_hilos   # Consistente con NUM_HILOS_OPENCV del proyecto
    opts.inter_op_num_threads = num_hilos
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    for path, nombre in [(_PATH_YOLO, "YOLO detector"), (_PATH_LCNET, "LCNet regressor")]:
        if not os.path.isfile(path):
            print(f"[DocDetector] Modelo no encontrado: {path}. Recayendo en OpenCV heuristico.")
            _carga_fallida = True
            return False

    try:
        _sess_yolo  = ort.InferenceSession(_PATH_YOLO,  sess_options=opts, providers=providers)
        _sess_lcnet = ort.InferenceSession(_PATH_LCNET, sess_options=opts, providers=providers)
        return True
    except Exception as e:
        print(f"[DocDetector] Error cargando modelos ONNX: {e}")
        _carga_fallida = True
        return False


# ---------------------------------------------------------------------------
# Preprocesado compartido: normalizacion ImageNet estandar
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _bgr_a_tensor(imagen_bgr, tamano):
    """
    Redimensiona la imagen BGR a (tamano x tamano), convierte a RGB float32,
    normaliza con media/std ImageNet y retorna tensor NCHW [1, 3, tamano, tamano].
    """
    import cv2
    resized = cv2.resize(imagen_bgr, (tamano, tamano), interpolation=cv2.INTER_LINEAR)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norma   = (rgb - _MEAN) / _STD
    return np.ascontiguousarray(norma.transpose(2, 0, 1)[np.newaxis])   # [1,3,H,W# ---------------------------------------------------------------------------
# Etapa 1: YOLO — detectar bounding-boxes candidatos del documento
# ---------------------------------------------------------------------------
_YOLO_INPUT_SIZE  = 960    # Shape del modelo: [1, 3, 960, 960]
_YOLO_CONF_UMBRAL = 0.08   # Confianza minima para aceptar una deteccion candidata


def _inferir_bboxes_candidatos_yolo(imagen_bgr, conf_umbral=_YOLO_CONF_UMBRAL):
    """
    Ejecuta el modelo YOLO y retorna una lista de bboxes candidatos [x1, y1, x2, y2]
    en pixeles absolutos, ordenados por confianza descendente.
    """
    H, W = imagen_bgr.shape[:2]
    tensor = _bgr_a_tensor(imagen_bgr, _YOLO_INPUT_SIZE)
    salida = _sess_yolo.run(None, {"images": tensor})[0][0]  # [5, 18900]

    conf = salida[4]
    indices = np.where(conf > conf_umbral)[0]
    if len(indices) == 0:
        return []

    s = float(_YOLO_INPUT_SIZE)
    boxes_raw = []
    for idx in indices:
        c = float(conf[idx])
        cx, cy, bw, bh = salida[0, idx], salida[1, idx], salida[2, idx], salida[3, idx]
        x1 = float(np.clip((cx - bw / 2) / s, 0.0, 1.0))
        y1 = float(np.clip((cy - bh / 2) / s, 0.0, 1.0))
        x2 = float(np.clip((cx + bw / 2) / s, 0.0, 1.0))
        y2 = float(np.clip((cy + bh / 2) / s, 0.0, 1.0))
        boxes_raw.append((c, int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))

    # Ordenar por confianza y quedarse con las mejores cajas diversas
    boxes_raw.sort(key=lambda b: b[0], reverse=True)
    cajas_evaluar = []
    for b in boxes_raw:
        c, x1, y1, x2, y2 = b
        duplicada = False
        for cb in cajas_evaluar:
            _, cx1, cy1, cx2, cy2 = cb
            if abs(x1 - cx1) < 40 and abs(y1 - cy1) < 40 and abs(x2 - cx2) < 40 and abs(y2 - cy2) < 40:
                duplicada = True
                break
        if not duplicada:
            cajas_evaluar.append(b)
        if len(cajas_evaluar) >= 5:
            break

    return cajas_evaluar


def _inferir_bbox_yolo(imagen_bgr):
    """
    Retorna el mejor bounding box de YOLO [x1, y1, x2, y2] normalizado [0..1]
    para compatibilidad hacia atras.
    """
    candidatos = _inferir_bboxes_candidatos_yolo(imagen_bgr)
    if not candidatos:
        return None
    H, W = imagen_bgr.shape[:2]
    _, x1, y1, x2, y2 = candidatos[0]
    return float(x1 / W), float(y1 / H), float(x2 / W), float(y2 / H)


# ---------------------------------------------------------------------------
# Etapa 2: LCNet — regresion de 4 esquinas via heatmaps
# ---------------------------------------------------------------------------
_LCNET_INPUT_SIZE   = 256   # Shape del modelo: [1, 3, 256, 256]
_LCNET_HEATMAP_SIZE = 128   # Shape del heatmap de salida: [1, 4, 128, 128]
_LCNET_CONF_UMBRAL  = 0.05  # Confianza minima del pico del heatmap


def _inferir_esquinas_lcnet(recorte_bgr):
    """
    Ejecuta el modelo LCNet sobre el recorte del documento y retorna las
    4 esquinas como array [4, 2] con coordenadas normalizadas [0..1] relativas
    al recorte entregado, con reconstruccion vectorial si falta 1 esquina.
    """
    tensor   = _bgr_a_tensor(recorte_bgr, _LCNET_INPUT_SIZE)
    heatmaps = _sess_lcnet.run(None, {"img": tensor})[0][0]  # [4, 128, 128]

    esquinas  = np.zeros((4, 2), dtype=np.float32)
    confianza = np.zeros(4, dtype=np.float32)
    N = _LCNET_HEATMAP_SIZE

    for k in range(4):
        hm      = heatmaps[k]
        confianza[k] = float(hm.max())

        peak_r, peak_c = np.unravel_index(np.argmax(hm), hm.shape)

        # Centroide ponderado en vecindario 9x9 para precision subpixel
        r1 = max(0,   peak_r - 4)
        r2 = min(N,   peak_r + 5)
        c1 = max(0,   peak_c - 4)
        c2 = min(N,   peak_c + 5)
        patch  = hm[r1:r2, c1:c2].astype(np.float64)
        total  = patch.sum()
        if total > 1e-9:
            rows_idx, cols_idx = np.indices(patch.shape)
            refined_r = (rows_idx * patch).sum() / total + r1
            refined_c = (cols_idx * patch).sum() / total + c1
        else:
            refined_r, refined_c = float(peak_r), float(peak_c)

        esquinas[k, 0] = np.clip(refined_c / (N - 1), 0.0, 1.0)
        esquinas[k, 1] = np.clip(refined_r / (N - 1), 0.0, 1.0)

    # Validar confianza de esquinas y reconstruccion adaptativa:
    sorted_confs = np.sort(confianza)[::-1]

    # Caso 1: 4 esquinas buenas
    if np.sum(confianza >= 0.12) == 4:
        return esquinas

    # Caso 2: 3 esquinas buenas con umbral adaptativo
    max_c = sorted_confs[0]
    umbral = max(0.08, min(0.15, max_c * 0.15))
    buenas = confianza >= umbral
    num_buenas = int(np.sum(buenas))

    # Caso 3: 2 esquinas excelentes (>= 0.70) y una 3ra solida (>= 0.07)
    if num_buenas < 3 and sorted_confs[0] >= 0.70 and sorted_confs[1] >= 0.65 and sorted_confs[2] >= 0.07:
        buenas = confianza >= 0.07
        num_buenas = int(np.sum(buenas))

    if num_buenas < 3:
        return None

    if num_buenas == 3:
        missing_idx = int(np.where(~buenas)[0][0])
        # Orden: 0: TL, 1: TR, 2: BR, 3: BL
        if missing_idx == 0:    # TL faltante
            esquinas[0] = esquinas[1] + esquinas[3] - esquinas[2]
        elif missing_idx == 1:  # TR faltante
            esquinas[1] = esquinas[0] + esquinas[2] - esquinas[3]
        elif missing_idx == 2:  # BR faltante
            esquinas[2] = esquinas[1] + esquinas[3] - esquinas[0]
        elif missing_idx == 3:  # BL faltante
            esquinas[3] = esquinas[0] + esquinas[2] - esquinas[1]

    return esquinas


# ---------------------------------------------------------------------------
# Validacion geometrica de las 4 esquinas
# ---------------------------------------------------------------------------
_MIN_AREA_CUADRILATERO = 0.05  # El poligono debe cubrir al menos 5% del recorte


def _validar_cuadrilatero(pts_norm, ancho, alto):
    """
    Verifica que las 4 esquinas formen un cuadrilatero convexo de tamano razonable
    y que las esquinas esten suficientemente separadas (no colapsadas en un punto).
    pts_norm: [4, 2] coordenadas normalizadas al recorte.
    """
    import cv2
    pts_px = (pts_norm * np.array([ancho, alto])).astype(np.float32)

    # Verificar que el hull convexo tenga exactamente 4 vertices (estricto cuadrilatero)
    hull = cv2.convexHull(pts_px)
    if len(hull) < 4:
        return False

    area = cv2.contourArea(hull)
    if area / (ancho * alto) < _MIN_AREA_CUADRILATERO:
        return False

    # Verificar que ninguna esquina este demasiado cerca de otra (< 5% del lado menor)
    min_lado = min(ancho, alto)
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(pts_px[i] - pts_px[j])
            if dist < 0.05 * min_lado:
                return False

    # Verificar aspect ratio del documento
    tl, tr, br, bl = pts_px
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    avg_w = (w_top + w_bot) / 2.0
    avg_h = (h_left + h_right) / 2.0
    if avg_w < 30 or avg_h < 30:
        return False
    ratio = max(avg_w / avg_h, avg_h / avg_w)
    if ratio > 2.5:
        return False

    # Filtro anti-distorsión: los lados opuestos de una hoja de papel deben ser casi paralelos
    diff_w = abs(w_top - w_bot) / max(w_top, w_bot)
    diff_h = abs(h_left - h_right) / max(h_left, h_right)
    if diff_w > 0.18 or diff_h > 0.18:
        return False

    # Divergencia angular entre bordes horizontales (top vs bottom)
    ang_t = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    ang_b = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    if abs(ang_t - ang_b) > 7.5:
        return False

    return True


# ---------------------------------------------------------------------------
# Evaluacion de cuadrilateros en una orientacion dada
# ---------------------------------------------------------------------------
def _evaluar_deteccion_en_orientacion(imagen_bgr):
    """
    Evalua las cajas candidatas de YOLO sobre imagen_bgr y obtiene las esquinas
    de LCNet con la mejor puntuacion global.
    """
    import cv2
    H, W = imagen_bgr.shape[:2]
    candidatos = _inferir_bboxes_candidatos_yolo(imagen_bgr)
    if not candidatos:
        candidatos = []
    # Evaluar siempre la imagen completa para que LCNet pueda inferir esquinas directas
    candidatos_lcnet = list(candidatos)
    candidatos_lcnet.append((0.50, 0, 0, W, H))

    mejores_opciones = []

    for c_yolo, x1, y1, x2, y2 in candidatos_lcnet:
        bw = x2 - x1
        bh = y2 - y1
        if bw < 70 or bh < 70:
            continue

        # Probar recorte exacto primero, y padding ligero de respaldo
        variantes = [(0, 0)]
        pad_x = int(bw * 0.03)
        pad_y = int(bh * 0.03)
        if pad_x > 0 or pad_y > 0:
            variantes.append((pad_x, pad_y))

        for px, py in variantes:
            rx1 = max(0, x1 - px)
            ry1 = max(0, y1 - py)
            rx2 = min(W, x2 + px)
            ry2 = min(H, y2 + py)

            recorte = imagen_bgr[ry1:ry2, rx1:rx2]
            rec_h, rec_w = recorte.shape[:2]
            if rec_w < 60 or rec_h < 60:
                continue

            esquinas_norm = _inferir_esquinas_lcnet(recorte)
            if esquinas_norm is None:
                continue

            if not _validar_cuadrilatero(esquinas_norm, rec_w, rec_h):
                continue

            esquinas_px = esquinas_norm * np.array([rec_w, rec_h], dtype=np.float32)
            esquinas_px[:, 0] += rx1
            esquinas_px[:, 1] += ry1

            area_pts = cv2.contourArea(esquinas_px)
            area_pct = area_pts / float(W * H)
            if area_pct < 0.15:
                continue

            tl, tr, br, bl = esquinas_px
            w_top = np.linalg.norm(tr - tl)
            w_bot = np.linalg.norm(br - bl)
            h_left = np.linalg.norm(bl - tl)
            h_right = np.linalg.norm(br - tr)
            w_avg = (w_top + w_bot) / 2.0
            h_avg = (h_left + h_right) / 2.0
            if w_avg < 50 or h_avg < 50:
                continue

            ar = h_avg / w_avg
            doc_ratio = max(ar, 1.0 / ar)

            # Medir regularidad geométrica del cuadrilátero (paralelismo)
            diff_w = abs(w_top - w_bot) / max(w_top, w_bot)
            diff_h = abs(h_left - h_right) / max(h_left, h_right)
            if diff_w > 0.18 or diff_h > 0.18:
                continue

            ang_t = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
            ang_b = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
            if abs(ang_t - ang_b) > 7.5:
                continue

            regularidad = max(0.0, 1.0 - (diff_w + diff_h) / 2.0)

            score = 2.5 + (c_yolo * 0.8) + (regularidad * 1.5)

            # Bonificacion simétrica por formato de documento estándar (A4/Carta vertical u horizontal)
            if 1.20 <= doc_ratio <= 1.95:
                score += 2.2
            elif 1.10 <= doc_ratio <= 2.20:
                score += 1.2

            # Penalizar asimetria marcada en lados opuestos (trapecios que arrastran fondos)
            if diff_w > 0.14 or diff_h > 0.14:
                score -= 1.8

            # Bonificación proporcional a la cobertura del documento (favorece la hoja completa vs fragmentos)
            score += (area_pct ** 1.5) * 4.0

            mejores_opciones.append({
                "esquinas_px": esquinas_px,
                "score": score,
                "ar": ar,
                "area_pct": area_pct
            })

    # Candidato ortogonal directo de YOLO:
    # Solo para detecciones reales de YOLO (excluyendo el dummy de imagen completa)
    for c_yolo, x1, y1, x2, y2 in candidatos:
        bw = x2 - x1
        bh = y2 - y1
        box_area_pct = float(bw * bh) / float(W * H)
        if c_yolo >= 0.15 and 0.25 <= box_area_pct <= 0.85:
            box_ar = float(bh) / float(bw)
            box_ratio = max(box_ar, 1.0 / box_ar)
            if 1.15 <= box_ratio <= 1.65:
                pts_bbox = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]
                ], dtype=np.float32)
                box_score = 3.0 + (c_yolo * 1.5)
                if 1.25 <= box_ratio <= 1.52:
                    box_score += 1.0
                box_score += (box_area_pct ** 1.5) * 4.0
                mejores_opciones.append({
                    "esquinas_px": pts_bbox,
                    "score": box_score,
                    "ar": box_ar,
                    "area_pct": box_area_pct
                })

    if not mejores_opciones:
        return None

    mejores_opciones.sort(key=lambda item: item["score"], reverse=True)
    return mejores_opciones[0]



# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def detectar_esquinas_documento(imagen_bgr):
    """
    Detecta las 4 esquinas del documento principal en imagen_bgr usando
    Deep Learning (ONNX) con analisis multi-orientacion y refinamiento subpixel.

    Retorna:
        np.ndarray de shape [4, 2] con las 4 esquinas [TL, TR, BR, BL] en
        pixeles absolutos de imagen_bgr (float32).
        None si la deteccion falla o no supera los umbrales de calidad.
    """
    if not _cargar_modelos():
        return None

    alto_orig, ancho_orig = imagen_bgr.shape[:2]

    try:
        # Evaluar sistemáticamente las 4 orientaciones para robustez total
        if ancho_orig > alto_orig:
            orientaciones = [
                (cv2.ROTATE_90_COUNTERCLOCKWISE, "90_CCW"),
                (None, "0_DEG"),
                (cv2.ROTATE_180, "180_DEG"),
                (cv2.ROTATE_90_CLOCKWISE, "90_CW")
            ]
        else:
            orientaciones = [
                (None, "0_DEG"),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, "90_CCW"),
                (cv2.ROTATE_180, "180_DEG"),
                (cv2.ROTATE_90_CLOCKWISE, "90_CW")
            ]

        todas_detecciones = []

        for rot_code, rot_name in orientaciones:
            im_rot = imagen_bgr if rot_code is None else cv2.rotate(imagen_bgr, rot_code)
            res = _evaluar_deteccion_en_orientacion(im_rot)
            if res is not None:
                todas_detecciones.append((res, rot_code))
                # Cortar anticipadamente solo si la detección tiene cobertura casi total y score superlativo
                if res["score"] >= 9.2 and res["area_pct"] >= 0.85:
                    break

        if not todas_detecciones:
            return None

        todas_detecciones.sort(key=lambda item: item[0]["score"], reverse=True)
        mejor_res, mejor_rot = todas_detecciones[0]

        pts_rot = mejor_res["esquinas_px"]
        if mejor_rot is None:
            pts_orig = pts_rot.copy()
        elif mejor_rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
            pts_orig = np.zeros_like(pts_rot)
            pts_orig[:, 0] = ancho_orig - 1 - pts_rot[:, 1]
            pts_orig[:, 1] = pts_rot[:, 0]
        elif mejor_rot == cv2.ROTATE_90_CLOCKWISE:
            pts_orig = np.zeros_like(pts_rot)
            pts_orig[:, 0] = pts_rot[:, 1]
            pts_orig[:, 1] = alto_orig - 1 - pts_rot[:, 0]
        elif mejor_rot == cv2.ROTATE_180:
            pts_orig = np.zeros_like(pts_rot)
            pts_orig[:, 0] = ancho_orig - 1 - pts_rot[:, 0]
            pts_orig[:, 1] = alto_orig - 1 - pts_rot[:, 1]

        # Ordenar canónicamente [TL, TR, BR, BL] para que la perspectiva siempre sea ortogonal y derecha
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts_orig.sum(axis=1)
        rect[0] = pts_orig[np.argmin(s)]  # Top-Left (suma x+y mínima)
        rect[2] = pts_orig[np.argmax(s)]  # Bottom-Right (suma x+y máxima)
        diff = np.diff(pts_orig, axis=1)
        rect[1] = pts_orig[np.argmin(diff)]  # Top-Right (diferencia y-x mínima)
        rect[3] = pts_orig[np.argmax(diff)]  # Bottom-Left (diferencia y-x máxima)

        return rect

    except Exception as e:
        print(f"[DocDetector] Error durante inferencia: {e}")
        return None

