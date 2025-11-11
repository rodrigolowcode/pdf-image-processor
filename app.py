"""
PDF/Image Processor - Production Ready v3.3
Configurável via variáveis de ambiente do Easypanel
Inclui detecção e limpeza opcional de linhas sobre texto
Cada request usa 2-3x o tamanho do arquivo em RAM
Ex: 30MB JPEG → ~90-120MB RAM. Configure workers no Gunicorn conforme RAM disponível.
"""

from flask import Flask, request, send_file, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pdf2image import convert_from_path
import cv2
import numpy as np
import os
import tempfile
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from functools import wraps
from typing import Tuple
import time
from io import BytesIO
import signal
import sys

# ============================================================================
# CONFIGURAÇÕES VIA ENVIRONMENT VARIABLES
# ============================================================================

CONFIG = {
    'MAX_CONTENT_LENGTH': int(os.getenv('MAX_CONTENT_LENGTH', 30 * 1024 * 1024)),
    'MIN_DIMENSION': int(os.getenv('MIN_DIMENSION', 1200)),
    'MAX_DIMENSION': int(os.getenv('MAX_DIMENSION', 3500)),
    'CLAHE_CLIP_LIMIT': float(os.getenv('CLAHE_CLIP_LIMIT', 1.8)),
    'CLAHE_TILE_SIZE': int(os.getenv('CLAHE_TILE_SIZE', 24)),
    'JPEG_QUALITY': int(os.getenv('JPEG_QUALITY', 92)),
    'PDF_DPI': int(os.getenv('PDF_DPI', 200)),
    'MEDIAN_KERNEL': int(os.getenv('MEDIAN_KERNEL', 3)),
    'NOISE_THRESHOLD': int(os.getenv('NOISE_THRESHOLD', 100)),
    'LOW_CONTRAST_THRESHOLD': int(os.getenv('LOW_CONTRAST_THRESHOLD', 50)),
    'MAGIC_BUFFER_SIZE': int(os.getenv('MAGIC_BUFFER_SIZE', 8192)),
    'ENABLE_LINE_CLEANUP': os.getenv('ENABLE_LINE_CLEANUP', 'false').lower() == 'true',
}

ALLOWED_MIMES = {
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/tiff',
    'image/bmp'
}

# ============================================================================
# REDIS CONFIGURATION
# ============================================================================

def get_redis_url():
    """
    Constrói Redis URL a partir de variáveis individuais ou usa URL completa
    """
    # Prioridade 1: REDIS_URL completa
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        return redis_url
    
    # Prioridade 2: Construir a partir de variáveis separadas
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')
    redis_password = os.getenv('REDIS_PASSWORD', '')
    redis_db = os.getenv('REDIS_DB', '0')
    
    # Construir URL
    if redis_password:
        return f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
    else:
        return f"redis://{redis_host}:{redis_port}/{redis_db}"

# ============================================================================
# INICIALIZAÇÃO FLASK
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']

# Redis URL configurável via ENV
redis_url = get_redis_url()

# Rate Limiter com Redis ou Memory
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{os.getenv('RATE_LIMIT_PER_HOUR', 200)} per hour"],
    storage_uri=redis_url,
    strategy="fixed-window"
)

# Logging configurável via ENV
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Log de inicialização
logger.info("=" * 80)
logger.info("🚀 PDF/Image Processor v3.3 iniciando")
logger.info(f"📊 Configurações:")
logger.info(f"   • Max file size: {CONFIG['MAX_CONTENT_LENGTH'] / (1024**2):.0f}MB")
logger.info(f"   • Min dimension: {CONFIG['MIN_DIMENSION']}px")
logger.info(f"   • Max dimension: {CONFIG['MAX_DIMENSION']}px")
logger.info(f"   • PDF DPI: {CONFIG['PDF_DPI']}")
logger.info(f"   • JPEG quality: {CONFIG['JPEG_QUALITY']}")
logger.info(f"   • Line cleanup: {'ENABLED' if CONFIG['ENABLE_LINE_CLEANUP'] else 'DISABLED'}")
logger.info(f"🔗 Redis: {redis_url}")
logger.info(f"🚦 Rate limits: {os.getenv('RATE_LIMIT_PER_MINUTE', 10)}/min, {os.getenv('RATE_LIMIT_PER_HOUR', 100)}/hour")
logger.info(f"📝 Log level: {log_level}")
logger.info("=" * 80)

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def handle_sigterm(*args):
    """Graceful shutdown"""
    logger.info("🛑 SIGTERM recebido - shutdown graceful")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

# ============================================================================
# UTILITÁRIOS
# ============================================================================

def timeit(func):
    """Decorator para monitorar performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱️  {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper

def secure_validation(file: FileStorage) -> Tuple[str, bytes, str]:
    """
    Valida arquivo ANTES de salvar em disco
    Retorna nome seguro, buffer validado e mime type
    """
    # 1. Validação de tamanho
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    
    if size > CONFIG['MAX_CONTENT_LENGTH']:
        raise ValueError(
            f"Arquivo muito grande: {size / (1024**2):.1f}MB. "
            f"Máximo: {CONFIG['MAX_CONTENT_LENGTH'] / (1024**2):.0f}MB"
        )
    
    if size == 0:
        raise ValueError("Arquivo vazio")
    
    # 2. Lê buffer completo para memória
    file.stream.seek(0)
    buffer = file.stream.read()
    
    # 3. Valida MIME type do buffer (8KB para segurança)
    try:
        import magic
        mime = magic.from_buffer(
            buffer[:CONFIG['MAGIC_BUFFER_SIZE']], 
            mime=True
        )
        logger.info(f"🔍 MIME detectado: {mime}")
    except ImportError:
        # Fallback para extensão se magic não disponível
        logger.warning("⚠️  python-magic não disponível, usando fallback")
        ext = Path(file.filename).suffix.lower()
        mime_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        mime = mime_map.get(ext)
    except Exception as e:
        logger.error(f"Erro ao detectar MIME: {e}")
        raise ValueError("Não foi possível detectar tipo do arquivo")
    
    if not mime or mime not in ALLOWED_MIMES:
        raise ValueError(f"MIME type não permitido: {mime}")
    
    # 4. Retorna nome seguro, buffer validado e MIME
    return secure_filename(file.filename), buffer, mime

# ============================================================================
# DETECÇÃO E LIMPEZA DE LINHAS SOBRE TEXTO
# ============================================================================

@timeit
def detect_text_line_overlap_production(image: np.ndarray) -> dict:
    """
    VERSÃO PRODUÇÃO v3.3
    Detecta linhas sobre texto de forma otimizada
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Downscale adaptativo
    if max(height, width) > 4000:
        scale = 0.4
    elif min(height, width) > 2000:
        scale = 0.5
    else:
        scale = 1.0
    
    gray_analysis = gray if scale == 1.0 else cv2.resize(
        gray, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA
    )
    
    # Detecção de texto
    blur = cv2.GaussianBlur(gray_analysis, (3, 3), 0)
    text_mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    # Early return se pouco texto
    text_density = np.sum(text_mask > 0) / text_mask.size
    if text_density < 0.01:
        logger.info(f"✅ Densidade de texto < 1% (scale={scale:.2f}), skip")
        return {
            'has_problem': False,
            'overlap_score': 0,
            'severity': 'none',
            'problem_lines': [],
            'scale_used': scale,
            'total_lines_detected': 0,
            'text_density': float(text_density)
        }
    
    # Dilata texto
    kernel_size = max(2, int(2 * scale))
    text_mask_expanded = cv2.dilate(
        text_mask, 
        np.ones((kernel_size, kernel_size), np.uint8)
    )
    
    # Detecção de linhas
    canny_low = max(30, int(40 * scale))
    canny_high = max(100, int(120 * scale))
    edges = cv2.Canny(gray_analysis, canny_low, canny_high)
    
    hough_threshold = max(50, int(60 * scale))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 
        hough_threshold,
        minLineLength=int(80 * scale),
        maxLineGap=int(10 * scale)
    )
    
    if lines is None:
        logger.info(f"✅ Nenhuma linha detectada (scale={scale:.2f})")
        return {
            'has_problem': False,
            'overlap_score': 0,
            'severity': 'none',
            'problem_lines': [],
            'scale_used': scale,
            'total_lines_detected': 0,
            'text_density': float(text_density)
        }
    
    # Filtra e ordena
    def line_length_squared(line):
        x1, y1, x2, y2 = line[0]
        return (x2 - x1)**2 + (y2 - y1)**2
    
    min_length_sq = (50 * scale)**2
    lines_filtered = [l for l in lines 
                      if line_length_squared(l) >= min_length_sq]
    
    if not lines_filtered:
        logger.info(f"✅ Apenas linhas curtas (scale={scale:.2f})")
        return {
            'has_problem': False,
            'overlap_score': 0,
            'severity': 'none',
            'problem_lines': [],
            'scale_used': scale,
            'total_lines_detected': 0,
            'text_density': float(text_density)
        }
    
    sorted_lines = sorted(
        lines_filtered,
        key=line_length_squared,
        reverse=True
    )
    
    # Verifica overlap
    overlap_score = 0
    problem_lines = []
    
    for line in sorted_lines[:min(50, len(sorted_lines))]:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt(line_length_squared(line))
        
        line_mask = np.zeros_like(text_mask_expanded)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
        
        overlap = cv2.bitwise_and(text_mask_expanded, line_mask)
        overlap_pixels = np.sum(overlap > 0)
        
        threshold = max(40, int(line_length * 0.08))
        
        if overlap_pixels > threshold:
            overlap_score += 1
            
            problem_lines.append({
                'coords': tuple(int(c / scale) for c in (x1, y1, x2, y2)),
                'overlap_pixels': int(overlap_pixels / (scale**2)),
                'length': line_length / scale,
                'threshold_used': int(threshold / scale)
            })
            
            if overlap_score > 5:
                break
    
    # Validação de coordenadas
    if scale < 1.0:
        for line_info in problem_lines:
            x1, y1, x2, y2 = line_info['coords']
            line_info['coords'] = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)),
                max(0, min(y2, height - 1))
            )
    
    # Decisão final
    has_problem = overlap_score > 2
    severity = ('none' if overlap_score <= 1 else
                'moderate' if overlap_score <= 3 else 'severe')
    
    if has_problem:
        logger.info(
            f"⚠️  Overlap: score={overlap_score}, severity={severity}, "
            f"scale={scale:.2f}, density={text_density:.1%}, "
            f"lines={len(problem_lines)}/{len(sorted_lines)}"
        )
    else:
        logger.info(
            f"✅ Sem overlap (score={overlap_score}, scale={scale:.2f}, "
            f"density={text_density:.1%})"
        )
    
    return {
        'has_problem': has_problem,
        'overlap_score': overlap_score,
        'severity': severity,
        'problem_lines': problem_lines,
        'scale_used': scale,
        'total_lines_detected': len(sorted_lines),
        'text_density': float(text_density)
    }


@timeit
def clean_lines_over_text_production(image: np.ndarray, detection_result: dict) -> np.ndarray:
    """
    VERSÃO PRODUÇÃO v3.3
    Remove linhas detectadas via inpainting
    """
    
    # Validação
    if not detection_result.get('problem_lines'):
        logger.warning("⚠️  Nenhuma linha para limpar")
        return image
    
    if not detection_result.get('has_problem'):
        logger.info("✅ has_problem=False, skip limpeza")
        return image
    
    height, width = image.shape[:2]
    
    # Cria máscara
    lines_mask = np.zeros((height, width), dtype=np.uint8)
    valid_lines = 0
    
    for line_info in detection_result['problem_lines']:
        x1, y1, x2, y2 = line_info['coords']
        
        # Clipa coordenadas
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Valida
        if x1 == x2 and y1 == y2:
            logger.warning(f"⚠️  Linha colapsou: {line_info['coords']}")
            continue
        
        # Thickness baseado no comprimento
        length = line_info['length']
        if length > 300:
            thickness = 5
        elif length > 150:
            thickness = 4
        else:
            thickness = 3
        
        cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness)
        valid_lines += 1
    
    if valid_lines == 0:
        logger.warning("⚠️  Nenhuma linha válida")
        return image
    
    # Dilata
    lines_mask = cv2.dilate(
        lines_mask,
        np.ones((2, 2), np.uint8),
        iterations=1
    )
    
    # Inpainting condicional
    severity = detection_result.get('severity', 'moderate')
    
    if severity == 'moderate':
        # BGR direto (rápido)
        radius = 3
        result = cv2.inpaint(
            image, lines_mask,
            inpaintRadius=radius,
            flags=cv2.INPAINT_TELEA
        )
        method = 'BGR'
    else:
        # LAB (qualidade)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        radius = 5
        l_clean = cv2.inpaint(
            l, lines_mask,
            inpaintRadius=radius,
            flags=cv2.INPAINT_TELEA
        )
        
        result = cv2.cvtColor(cv2.merge([l_clean, a, b]), cv2.COLOR_LAB2BGR)
        method = 'LAB'
    
    pixels_cleaned = np.sum(lines_mask > 0)
    logger.info(
        f"🧹 Limpeza: {valid_lines} linhas, {pixels_cleaned}px, "
        f"method={method}, radius={radius}"
    )
    
    return result

# ============================================================================
# PROCESSAMENTO DE IMAGENS
# ============================================================================

@timeit
def optimize_for_cpu(image: np.ndarray) -> np.ndarray:
    """
    Pipeline otimizado v3.3
    Ordem: CLAHE → Upscale/Downscale → [Limpeza] → Denoise → Sharpen
    """
    
    # 1. CLAHE
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < CONFIG['LOW_CONTRAST_THRESHOLD']:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(
            clipLimit=CONFIG['CLAHE_CLIP_LIMIT'],
            tileGridSize=(CONFIG['CLAHE_TILE_SIZE'], CONFIG['CLAHE_TILE_SIZE'])
        )
        l = clahe.apply(l)
        
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        logger.info(f"✨ CLAHE aplicado (var: {laplacian_var:.2f})")
    else:
        logger.info(f"✅ Bom contraste (var: {laplacian_var:.2f})")
    
    # 2. UPSCALING/DOWNSCALING
    height, width = image.shape[:2]
    min_dim = min(height, width)
    max_dim = max(height, width)
    
    if min_dim < CONFIG['MIN_DIMENSION']:
        scale = CONFIG['MIN_DIMENSION'] / min_dim
        new_width = int(min(width * scale, CONFIG['MAX_DIMENSION']))
        new_height = int(min(height * scale, CONFIG['MAX_DIMENSION']))
        
        image = cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )
        logger.info(f"📐 Upscaled: {new_width}x{new_height}")
        
    elif max_dim > CONFIG['MAX_DIMENSION']:
        scale = CONFIG['MAX_DIMENSION'] / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        image = cv2.resize(
            image, (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        logger.info(f"📉 Downscaled: {new_width}x{new_height}")
    
    # 3. DETECÇÃO E LIMPEZA (OPCIONAL)
    if CONFIG['ENABLE_LINE_CLEANUP']:
        detection = detect_text_line_overlap_production(image)
        
        if detection['has_problem']:
            logger.info(
                f"🔧 Limpeza: score={detection['overlap_score']}, "
                f"severity={detection['severity']}"
            )
            image = clean_lines_over_text_production(image, detection)
        else:
            logger.info("✅ Sem linhas problemáticas")
    
    # 4. DENOISING
    if laplacian_var < CONFIG['NOISE_THRESHOLD']:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.medianBlur(l, CONFIG['MEDIAN_KERNEL'])
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        logger.info(f"🧹 MedianBlur aplicado no canal L")
    
    # 5. SHARPENING
    blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
    sharpening_amount = 0.8
    image = cv2.addWeighted(
        image, 1.0 + sharpening_amount,
        blurred, -sharpening_amount,
        0
    )
    logger.info("🔪 Sharpening aplicado")
    
    return image

@timeit
def convert_pdf_to_image(pdf_buffer: bytes) -> np.ndarray:
    """Converte PDF buffer para imagem"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_buffer)
        tmp_path = tmp.name
    
    try:
        images = convert_from_path(
            tmp_path, dpi=CONFIG['PDF_DPI'],
            first_page=1, last_page=1,
            fmt='jpeg', thread_count=2,
            use_pdftocairo=True, grayscale=False
        )
        
        if not images:
            raise ValueError("PDF vazio ou corrompido")
        
        image_array = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        logger.info(f"📄 PDF convertido: {image_array.shape}")
        return image_array
        
    finally:
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logger.warning(f"Falha ao remover temp PDF: {e}")

@timeit
def process_to_memory(buffer: bytes, mime_type: str) -> BytesIO:
    """Processa arquivo em memória"""
    if mime_type == 'application/pdf':
        image = convert_pdf_to_image(buffer)
    else:
        nparr = np.frombuffer(buffer, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Não foi possível decodificar imagem")
    
    height, width = image.shape[:2]
    if max(height, width) > CONFIG['MAX_DIMENSION']:
        raise ValueError(
            f"Dimensões muito grandes: {width}x{height}px. "
            f"Máximo: {CONFIG['MAX_DIMENSION']}px"
        )
    
    logger.info(f"📊 Original: {width}x{height}")
    
    processed = optimize_for_cpu(image)
    
    success, buffer_encoded = cv2.imencode(
        '.jpg', processed,
        [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']]
    )
    
    if not success:
        raise ValueError("Falha ao codificar imagem")
    
    output_buffer = BytesIO(buffer_encoded.tobytes())
    output_buffer.seek(0)
    
    file_size = len(buffer_encoded) / 1024
    logger.info(f"✅ Processado: {file_size:.1f}KB")
    
    return output_buffer

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'Arquivo muito grande',
        'max_size_mb': CONFIG['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    }), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit excedido',
        'message': str(e.description)
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erro 500: {error}", exc_info=True)
    return jsonify({'error': 'Erro interno do servidor'}), 500

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'pdf-image-processor',
        'version': '3.3',
        'worker_pid': os.getpid(),
        'redis': redis_url,
        'config': {
            'max_file_mb': CONFIG['MAX_CONTENT_LENGTH'] / (1024 * 1024),
            'min_dimension': CONFIG['MIN_DIMENSION'],
            'max_dimension': CONFIG['MAX_DIMENSION'],
            'line_cleanup_enabled': CONFIG['ENABLE_LINE_CLEANUP']
        }
    }), 200

@app.route('/ready', methods=['GET'])
def readiness_check():
    return jsonify({'status': 'ready'}), 200

@app.route('/process', methods=['POST'])
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_MINUTE', 10)} per minute")
@limiter.limit(f"{os.getenv('RATE_LIMIT_PER_HOUR', 100)} per hour")
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if not file.filename:
        return jsonify({'error': 'Nome de arquivo vazio'}), 400
    
    try:
        filename, buffer, mime_type = secure_validation(file)
        
        logger.info(
            f"🔄 Processando: {filename} "
            f"({len(buffer) / 1024:.1f}KB, {mime_type})"
        )
        
        output_buffer = process_to_memory(buffer, mime_type)
        
        return send_file(
            output_buffer,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f"processed_{Path(filename).stem}.jpg"
        )
        
    except ValueError as e:
        logger.warning(f"⚠️  Validação: {e}")
        return jsonify({'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
        return jsonify({'error': 'Erro ao processar arquivo'}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'PDF/Image Processor',
        'version': '3.3',
        'status': 'production-ready',
        'worker_pid': os.getpid(),
        'improvements_v3.3': [
            'Detecção inteligente de linhas sobre texto',
            'Limpeza opcional via ENABLE_LINE_CLEANUP',
            'Performance: 2.5x mais rápido com downscale adaptativo',
            'Early return por densidade de texto',
            'Inpainting condicional (BGR ou LAB)'
        ],
        'endpoints': {
            'POST /process': {
                'description': 'Processa PDF ou imagem',
                'rate_limit': f"{os.getenv('RATE_LIMIT_PER_MINUTE', 10)}/min, {os.getenv('RATE_LIMIT_PER_HOUR', 100)}/hora",
                'max_size_mb': CONFIG['MAX_CONTENT_LENGTH'] / (1024 * 1024),
                'formats': list(ALLOWED_MIMES)
            },
            'GET /health': 'Health check',
            'GET /ready': 'Readiness check'
        },
        'config': {
            'min_dimension': f"{CONFIG['MIN_DIMENSION']}px",
            'max_dimension': f"{CONFIG['MAX_DIMENSION']}px",
            'pdf_dpi': CONFIG['PDF_DPI'],
            'jpeg_quality': CONFIG['JPEG_QUALITY'],
            'line_cleanup': 'ENABLED' if CONFIG['ENABLE_LINE_CLEANUP'] else 'DISABLED'
        },
        'environment': {
            'redis_url': redis_url,
            'log_level': log_level,
            'configurable_via': [
                'MAX_CONTENT_LENGTH', 'MIN_DIMENSION', 'MAX_DIMENSION',
                'PDF_DPI', 'JPEG_QUALITY', 'ENABLE_LINE_CLEANUP',
                'REDIS_URL', 'RATE_LIMIT_PER_MINUTE', 'RATE_LIMIT_PER_HOUR',
                'GUNICORN_WORKERS', 'GUNICORN_TIMEOUT'
            ]
        }
    }), 200

# ============================================================================
# GUNICORN HOOKS
# ============================================================================

def on_starting(server):
    logger.info("🚀 Gunicorn iniciando")

def when_ready(server):
    logger.info("✅ Servidor pronto")

def worker_int(worker):
    logger.info(f"⚠️  Worker {worker.pid} SIGINT")

def worker_abort(worker):
    logger.error(f"❌ Worker {worker.pid} abortado")

def pre_fork(server, worker):
    logger.info(f"🔄 Criando worker")

def post_fork(server, worker):
    logger.info(f"✅ Worker {worker.pid} iniciado")

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == '__main__':
    logger.warning("⚠️  MODO DESENVOLVIMENTO - Use Gunicorn em produção!")
    logger.info("🚀 Servidor: http://0.0.0.0:8000")
    logger.info("📖 Produção: gunicorn --config gunicorn.conf.py app:app")
    logger.info(f"💾 RAM/request: ~{CONFIG['MAX_CONTENT_LENGTH'] * 3 / (1024**2):.0f}MB")
    logger.info(f"🔧 Line cleanup: {'ENABLED' if CONFIG['ENABLE_LINE_CLEANUP'] else 'DISABLED'}")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
