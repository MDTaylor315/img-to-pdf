# ==============================================================================
# CONFIGURACIÓN DEL SISTEMA DE ESCANEO DE IMÁGENES A PDF (MontanoImagen)
# Modifica estos valores para controlar la calidad y la carga del servidor.
# ==============================================================================

# --- RENDIMIENTO Y CONTROL DE CPU EN HORA PUNTA ---
# Número de hilos de CPU que puede usar OpenCV por proceso.
# RECOMENDACIÓN EN HORA PUNTA: Setear a 1 para evitar saturación de vCPUs concurrentes.
NUM_HILOS_OPENCV = 1

# Dimensión máxima en píxeles (ancho o alto) para redimensionar la imagen HD.
# 2500 px = Calidad súper HD >300 DPI.
# TIP HORA PUNTA: Si el servidor recibe muchas peticiones simultáneas, 
# puedes reducirlo a 1800 px para acelerar el procesamiento un 40% adicional.
MAX_DIM_IMAGEN = 2500

# Dimensión máxima de la miniatura para análisis rápido de contornos.
MAX_DIM_MINIATURA_ANALISIS = 800


# --- CONTROL DE MEMORIA RAM ---
# Fuerza la liberación explícita de memoria RAM (Garbage Collection) tras procesar cada foto.
# Mantiene el consumo de RAM plano (~15-25 MB por usuario) sin importar si son 10 o 200 fotos.
LIMPIAR_RAM_POR_PAGINA = True


# --- PARAMETROS DE CALIDAD Y COMPRESIÓN ---
# Calidad de compresión JPEG guardado en RAM (1 a 100).
# 84 - 88: Balance perfecto de nitidez en firmas/texto con bajo peso (~200 KB por hoja).
CALIDAD_JPEG = 88

# Modo de procesamiento por defecto:
# - "magico": Escáner HD profesional (fondo blanco pulcro, texto y sellos oscuros).
# - "otsu": Blanco y Negro puro (1-bit).
MODO_PROCESAMIENTO_DEFECTO = "magico"

# Activar o desactivar recorte automático de perspectiva por defecto.
USAR_AUTO_CROP_DEFECTO = True

# Umbral mínimo de cobertura de papel para justificar auto-crop (0.08 = 8% de la foto).
# Bajarlo permite recortar fotos donde el papel ocupa poco (papel sobre escritorio).
PORCENTAJE_MIN_COBERTURA_PAPEL = 0.08

# Cobertura máxima: si el candidato ocupa casi toda la foto no hay nada que recortar.
PORCENTAJE_MAX_COBERTURA_PAPEL = 0.985

# Relación de aspecto aceptada para la hoja detectada (ancho / alto).
RELACION_ASPECTO_MIN = 0.25
RELACION_ASPECTO_MAX = 4.0

# Solidez mínima: área del cuadrilátero respecto al rectángulo rotado que lo contiene.
# Descarta formas irregulares (sombras, manchas del escritorio) que no son una hoja.
SOLIDEZ_MIN_CUADRILATERO = 0.80

# Margen de seguridad añadido alrededor de la hoja detectada (0.01 = 1% del tamaño).
# Evita cortar cabeceras o el borde mismo del papel.
MARGEN_EXTRA_RECORTE = 0.01


# --- AJUSTE FINO DE IMAGEN (FILTRO MÁGICO) ---
# Límite de corte para ecualización CLAHE (1.0 a 3.0).
CLIP_LIMIT_CLAHE = 1.5

# Fuerza del filtro de nitidez / unsharp mask (1.0 a 2.0).
FUERZA_NITIDEZ = 1.2
