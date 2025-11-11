"""
PDF/Image Processor - Production Ready v3.4
API Key obrigatória + Rate Limiting avançado por IP
Configurável via variáveis de ambiente do Easypanel
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
import secrets

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
    
    # DETECÇÃO DE LINHAS
    'ENABLE_LINE_CLEANUP': os.getenv('ENABLE_LINE_CLEANUP', 'false').lower() == 'true',
    'LINE_OVERLAP_THRESHOLD': int(os.getenv('LINE_OVERLAP_THRESHOLD', 2)),
    'LINE_MIN_TEXT_DENSITY': float(os.getenv('LINE_MIN_TEXT_DENSITY', 0.01)),
    'LINE_MIN_PIXELS_OVERLAP': int(os.getenv('LINE_MIN_PIXELS_OVERLAP', 40)),
    'LINE_OVERLAP_PERCENTAGE': float(os.getenv('LINE_OVERLAP_PERCENTAGE', 0.08)),
    'LINE_MIN_LENGTH': int(os.getenv('LINE_MIN_LENGTH', 50)),
    'LINE_HOUGH_THRESHOLD': int(os.getenv('LINE_HOUGH_THRESHOLD', 60)),
    'LINE_HOUGH_MIN_LENGTH': int(os.getenv('LINE_HOUGH_MIN_LENGTH', 80)),
    'LINE_HOUGH_MAX_GAP': int(os.getenv('LINE_HOUGH_MAX_GAP', 10)),
    'LINE_CANNY_LOW': int(os.getenv('LINE_CANNY_LOW', 40)),
    'LINE_CANNY_HIGH': int(os.getenv('LINE_CANNY_HIGH', 120)),
    
    # SEGURANÇA (API KEY OBRIGATÓRIA)
    'API_KEYS': [k.strip() for k in os.getenv('API_KEYS', '').split(',') if k.strip()],
    'API_KEY_HEADER': os.getenv('API_KEY_HEADER', 'X-API-Key'),
    'RATE_LIMIT_PER_MINUTE': int(os.getenv('RATE_LIMIT_PER_MINUTE', 10)),
    'RATE_LIMIT_PER_HOUR': int(os.getenv('RATE_LIMIT_PER_HOUR', 100)),
    'RATE_LIMIT_PER_DAY': int(os.getenv('RATE_LIMIT_PER_DAY', 1000)),
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
    """Constrói Redis URL"""
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        return redis_url
    
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = os.getenv('REDIS_PORT', '6379')
    redis_password = os.getenv('REDIS_PASSWORD', '')
    redis_db = os.getenv('REDIS_DB', '0')
    
    if redis_password:
        return f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
    else:
        return f"redis://{redis_host}:{redis_port}/{redis_db}"

# ============================================================================
# INICIALIZAÇÃO FLASK
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']

redis_url = get_redis_url()

# Rate Limiter com limites configuráveis
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[
        f"{CONFIG['RATE_LIMIT_PER_MINUTE']} per minute",
        f"{CONFIG['RATE_LIMIT_PER_HOUR']} per hour",
        f"{CONFIG['RATE_LIMIT_PER_DAY']} per day"
    ],
    storage_uri=redis_url,
    strategy="fixed-window",
    headers_enabled=True
)

log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Log de inicialização
logger.info("=" * 80)
logger.info("🚀 PDF/Image Processor v3.4 iniciando")
logger.info(f"📊 Configurações:")
logger.info(f"   • Max file size: {CONFIG['MAX_CONTENT_LENGTH'] / (1024**2):.0f}MB")
logger.info(f"   • Min dimension: {CONFIG['MIN_DIMENSION']}px")
logger.info(f"   • Max dimension: {CONFIG['MAX_DIMENSION']}px")
logger.info(f"   • PDF DPI: {CONFIG['PDF_DPI']}")
logger.info(f"   • JPEG quality: {CONFIG['JPEG_QUALITY']}")

if CONFIG['ENABLE_LINE_CLEANUP']:
    logger.info(f"🔧 Detecção de linhas: HABILITADA")
    logger.info(f"   • Overlap threshold: {CONFIG['LINE_OVERLAP_THRESHOLD']}")
    logger.info(f"   • Text density min: {CONFIG['LINE_MIN_TEXT_DENSITY']:.3f}")
else:
    logger.info(f"🔧 Detecção de linhas: DESABILITADA")

logger.info(f"🔐 API Key: OBRIGATÓRIA")
logger.info(f"   • Keys configuradas: {len(CONFIG['API_KEYS'])}")
logger.info(f"   • Header: {CONFIG['API_KEY_HEADER']}")

if not CONFIG['API_KEYS']:
    logger.error("❌ NENHUMA API KEY CONFIGURADA! Serviço não funcionará.")
    logger.error("   Configure API_KEYS no Easypanel")

logger.info(f"🔗 Redis: {redis_url}")
logger.info(f"🚦 Rate limits: {CONFIG['RATE_LIMIT_PER_MINUTE']}/min, {CONFIG['RATE_LIMIT_PER_HOUR']}/hour, {CONFIG['RATE_LIMIT_PER_DAY']}/day")
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
# SEGURANÇA - API KEY (OBRIGATÓRIA)
# ============================================================================

def require_api_key(f):
    """
    Decorator para proteger endpoints com API Key (OBRIGATÓRIA)
    Verifica header X-API-Key (ou configurado via API_KEY_HEADER)
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # API Key sempre obrigatória
        if not CONFIG['API_KEYS']:
            logger.error("🚫 Nenhuma API Key configurada no servidor")
            return jsonify({
                'error': 'Serviço não configurado',
                'message': 'Servidor sem API Keys configuradas'
            }), 503
        
        # Obtém API key do header
        api_key = request.headers.get(CONFIG['API_KEY_HEADER'])
        
        if not api_key:
            logger.warning(
                f"⚠️  Tentativa sem API Key de {get_remote_address()} "
                f"para {request.path}"
            )
            return jsonify({
                'error': 'API Key obrigatória',
                'message': f'Forneça a API Key no header {CONFIG["API_KEY_HEADER"]}'
            }), 401
        
        # Valida API key (constant-time comparison)
        valid = False
        for valid_key in CONFIG['API_KEYS']:
            if secrets.compare_digest(api_key, valid_key):
                valid = True
                break
        
        if not valid:
            logger.warning(
                f"🚫 API Key inválida de {get_remote_address()} "
                f"para {request.path}: {api_key[:8]}..."
            )
            return jsonify({
                'error': 'API Key inválida',
                'message': 'A API Key fornecida não é válida'
            }), 403
        
        # API Key válida
        logger.debug(f"✅ API Key válida de {get_remote_address()}")
        return f(*args, **kwargs)
    
    return decorated_function

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

def generate_api_key() -> str:
    """Gera uma API key segura de 32 bytes (64 caracteres hex)"""
    return secrets.token_hex(32)

def secure_validation(file: FileStorage) -> Tuple[str, bytes, str]:
    """Valida arquivo ANTES de salvar"""
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    
    if size > CONFIG['MAX_CONTENT_LENGTH']:
        raise ValueError(
            f"Arquivo muito grande: {size / (1024**2):.1f}MB. "
            f"Máximo: {CONFIG['MAX_CONTENT_LENGTH'] / (1024**2):.0f}MB"
        )
    
    if size == 0:
        raise ValueError("Arquivo vazio")
    
    file.stream.seek(0)
    buffer = file.stream.read()
    
    try:
        import magic
        mime = magic.from_buffer(buffer[:CONFIG['MAGIC_BUFFER_SIZE']], mime=True)
        logger.info(f"🔍 MIME detectado: {mime}")
    except ImportError:
        logger.warning("⚠️  python-magic não disponível, usando fallback")
        ext = Path(file.filename).suffix.lower()
        mime_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tiff': 'image/tiff', '.tif': 'image/tiff',
            '.bmp': 'image/bmp'
        }
        mime = mime_map.get(ext)
    except Exception as e:
        logger.error(f"Erro ao detectar MIME: {e}")
        raise ValueError("Não foi possível detectar tipo do arquivo")
    
    if not mime or mime not in ALLOWED_MIMES:
        raise ValueError(f"MIME type não permitido: {mime}")
    
    return secure_filename(file.filename), buffer, mime

# ============================================================================
# DETECÇÃO E LIMPEZA DE LINHAS (código mantido do v3.3)
# ============================================================================

@timeit
def detect_text_line_overlap_production(image: np.ndarray) -> dict:
    """Detecta linhas sobre texto"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    if max(height, width) > 4000:
        scale = 0.4
    elif min(height, width) > 2000:
        scale = 0.5
    else:
        scale = 1.0
    
    gray_analysis = gray if scale == 1.0 else cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    
    blur = cv2.GaussianBlur(gray_analysis, (3, 3), 0)
    text_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    text_density = np.sum(text_mask > 0) / text_mask.size
    if text_density < CONFIG['LINE_MIN_TEXT_DENSITY']:
        logger.info(f"✅ Densidade {text_density:.1%} < {CONFIG['LINE_MIN_TEXT_DENSITY']:.1%}, skip")
        return {
            'has_problem': False, 'overlap_score': 0, 'severity': 'none',
            'problem_lines': [], 'scale_used': scale,
            'total_lines_detected': 0, 'text_density': float(text_density)
        }
    
    kernel_size = max(2, int(2 * scale))
    text_mask_expanded = cv2.dilate(text_mask, np.ones((kernel_size, kernel_size), np.uint8))
    
    canny_low = max(30, int(CONFIG['LINE_CANNY_LOW'] * scale))
    canny_high = max(100, int(CONFIG['LINE_CANNY_HIGH'] * scale))
    edges = cv2.Canny(gray_analysis, canny_low, canny_high)
    
    hough_threshold = max(50, int(CONFIG['LINE_HOUGH_THRESHOLD'] * scale))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, hough_threshold,
        minLineLength=int(CONFIG['LINE_HOUGH_MIN_LENGTH'] * scale),
        maxLineGap=int(CONFIG['LINE_HOUGH_MAX_GAP'] * scale)
    )
    
    if lines is None:
        return {
            'has_problem': False, 'overlap_score': 0, 'severity': 'none',
            'problem_lines': [], 'scale_used': scale,
            'total_lines_detected': 0, 'text_density': float(text_density)
        }
    
    def line_length_squared(line):
        x1, y1, x2, y2 = line[0]
        return (x2 - x1)**2 + (y2 - y1)**2
    
    min_length_sq = (CONFIG['LINE_MIN_LENGTH'] * scale)**2
    lines_filtered = [l for l in lines if line_length_squared(l) >= min_length_sq]
    
    if not lines_filtered:
        return {
            'has_problem': False, 'overlap_score': 0, 'severity': 'none',
            'problem_lines': [], 'scale_used': scale,
            'total_lines_detected': 0, 'text_density': float(text_density)
        }
    
    sorted_lines = sorted(lines_filtered, key=line_length_squared, reverse=True)
    
    overlap_score = 0
    problem_lines = []
    
    for line in sorted_lines[:min(50, len(sorted_lines))]:
        x1, y1, x2, y2 = line[0]
        line_length = np.sqrt(line_length_squared(line))
        
        line_mask = np.zeros_like(text_mask_expanded)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
        
        overlap = cv2.bitwise_and(text_mask_expanded, line_mask)
        overlap_pixels = np.sum(overlap > 0)
        
        threshold = max(
            CONFIG['LINE_MIN_PIXELS_OVERLAP'],
            int(line_length * CONFIG['LINE_OVERLAP_PERCENTAGE'])
        )
        
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
    
    if scale < 1.0:
        for line_info in problem_lines:
            x1, y1, x2, y2 = line_info['coords']
            line_info['coords'] = (
                max(0, min(x1, width - 1)), max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)), max(0, min(y2, height - 1))
            )
    
    has_problem = overlap_score > CONFIG['LINE_OVERLAP_THRESHOLD']
    severity = ('none' if overlap_score <= 1 else
                'moderate' if overlap_score <= 3 else 'severe')
    
    if has_problem:
        logger.info(f"⚠️  Overlap: score={overlap_score}, severity={severity}")
    
    return {
        'has_problem': has_problem, 'overlap_score': overlap_score,
        'severity': severity, 'problem_lines': problem_lines,
        'scale_used': scale, 'total_lines_detected': len(sorted_lines),
        'text_density': float(text_density)
    }


@timeit
def clean_lines_over_text_production(image: np.ndarray, detection_result: dict) -> np.ndarray:
    """Remove linhas detectadas via inpainting"""
    if not detection_result.get('problem_lines') or not detection_result.get('has_problem'):
        return image
    
    height, width = image.shape[:2]
    lines_mask = np.zeros((height, width), dtype=np.uint8)
    valid_lines = 0
    
    for line_info in detection_result['problem_lines']:
        x1, y1, x2, y2 = line_info['coords']
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        if x1 == x2 and y1 == y2:
            continue
        
        length = line_info['length']
        thickness = 5 if length > 300 else 4 if length > 150 else 3
        cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness)
        valid_lines += 1
    
    if valid_lines == 0:
        return image
    
    lines_mask = cv2.dilate(lines_mask, np.ones((2, 2), np.uint8), iterations=1)
    
    severity = detection_result.get('severity', 'moderate')
    if severity == 'moderate':
        result = cv2.inpaint(image, lines_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        method = 'BGR'
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clean = cv2.inpaint(l, lines_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        result = cv2.cvtColor(cv2.merge([l_clean, a, b]), cv2.COLOR_LAB2BGR)
        method = 'LAB'
    
    logger.info(f"🧹 Limpeza: {valid_lines} linhas, method={method}")
    return result

# ============================================================================
# PROCESSAMENTO DE IMAGENS (código mantido do v3.3)
# ============================================================================

@timeit
def optimize_for_cpu(image: np.ndarray) -> np.ndarray:
    """Pipeline otimizado v3.4"""
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
    
    height, width = image.shape[:2]
    min_dim = min(height, width)
    max_dim = max(height, width)
    
    if min_dim < CONFIG['MIN_DIMENSION']:
        scale = CONFIG['MIN_DIMENSION'] / min_dim
        new_width = int(min(width * scale, CONFIG['MAX_DIMENSION']))
        new_height = int(min(height * scale, CONFIG['MAX_DIMENSION']))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        logger.info(f"📐 Upscaled: {new_width}x{new_height}")
    elif max_dim > CONFIG['MAX_DIMENSION']:
        scale = CONFIG['MAX_DIMENSION'] / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"📉 Downscaled: {new_width}x{new_height}")
    
    if CONFIG['ENABLE_LINE_CLEANUP']:
        detection = detect_text_line_overlap_production(image)
        if detection['has_problem']:
            image = clean_lines_over_text_production(image, detection)
    
    if laplacian_var < CONFIG['NOISE_THRESHOLD']:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.medianBlur(l, CONFIG['MEDIAN_KERNEL'])
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        logger.info("🧹 MedianBlur aplicado")
    
    blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
    image = cv2.addWeighted(image, 1.8, blurred, -0.8, 0)
    logger.info("🔪 Sharpening aplicado")
    
    return image

@timeit
def convert_pdf_to_image(pdf_buffer: bytes) -> np.ndarray:
    """Converte PDF para imagem"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_buffer)
        tmp_path = tmp.name
    
    try:
        images = convert_from_path(
            tmp_path, dpi=CONFIG['PDF_DPI'], first_page=1, last_page=1,
            fmt='jpeg', thread_count=2, use_pdftocairo=True, grayscale=False
        )
        if not images:
            raise ValueError("PDF vazio ou corrompido")
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

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
    logger.info(f"📊 Original: {width}x{height}")
    
    processed = optimize_for_cpu(image)
    
    success, buffer_encoded = cv2.imencode(
        '.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['JPEG_QUALITY']]
    )
    
    if not success:
        raise ValueError("Falha ao codificar imagem")
    
    output_buffer = BytesIO(buffer_encoded.tobytes())
    output_buffer.seek(0)
    
    logger.info(f"✅ Processado: {len(buffer_encoded) / 1024:.1f}KB")
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
    """Health check - SEM autenticação"""
    return jsonify({
        'status': 'healthy',
        'service': 'pdf-image-processor',
        'version': '3.4',
        'worker_pid': os.getpid(),
        'redis': redis_url,
        'security': {
            'api_key_required': True,
            'api_keys_configured': len(CONFIG['API_KEYS']),
            'api_key_header': CONFIG['API_KEY_HEADER']
        },
        'config': {
            'max_file_mb': CONFIG['MAX_CONTENT_LENGTH'] / (1024 * 1024),
            'line_cleanup_enabled': CONFIG['ENABLE_LINE_CLEANUP']
        }
    }), 200

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check - SEM autenticação"""
    return jsonify({'status': 'ready'}), 200

@app.route('/generate-key', methods=['POST'])
@require_api_key
def generate_key():
    """Gera nova API key (protegido - só admin pode gerar)"""
    new_key = generate_api_key()
    return jsonify({
        'api_key': new_key,
        'message': f'Adicione esta key em API_KEYS no Easypanel',
        'example': f'API_KEYS={CONFIG["API_KEYS"][0] if CONFIG["API_KEYS"] else ""},{new_key}'
    }), 200

@app.route('/process', methods=['POST'])
@require_api_key  # PROTEGIDO por API Key
@limiter.limit(
    f"{CONFIG['RATE_LIMIT_PER_MINUTE']} per minute;"
    f"{CONFIG['RATE_LIMIT_PER_HOUR']} per hour;"
    f"{CONFIG['RATE_LIMIT_PER_DAY']} per day"
)
def process_file():
    """Endpoint principal - REQUER API KEY"""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if not file.filename:
        return jsonify({'error': 'Nome de arquivo vazio'}), 400
    
    try:
        filename, buffer, mime_type = secure_validation(file)
        
        logger.info(
            f"🔄 Processando: {filename} ({len(buffer) / 1024:.1f}KB, {mime_type}) "
            f"de {get_remote_address()}"
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
    """Documentação - SEM autenticação"""
    return jsonify({
        'service': 'PDF/Image Processor',
        'version': '3.4',
        'status': 'production-ready',
        'improvements_v3.4': [
            '🔐 API Key obrigatória (sempre habilitada)',
            '🚦 Rate limiting avançado por IP (min/hora/dia)',
            '📊 Headers X-RateLimit-* informativos',
            '🔑 Endpoint /generate-key para criar novas keys',
            '⚡ Detecção de linhas configurável',
            '🛡️ Constant-time key comparison (timing attack protection)'
        ],
        'security': {
            'api_key_required': True,
            'api_key_header': CONFIG['API_KEY_HEADER'],
            'api_keys_configured': len(CONFIG['API_KEYS']),
            'rate_limits': {
                'per_minute': CONFIG['RATE_LIMIT_PER_MINUTE'],
                'per_hour': CONFIG['RATE_LIMIT_PER_HOUR'],
                'per_day': CONFIG['RATE_LIMIT_PER_DAY']
            }
        },
        'endpoints': {
            'POST /process': {
                'description': 'Processa PDF ou imagem',
                'authentication': f'API Key no header {CONFIG["API_KEY_HEADER"]}',
                'rate_limit': f"{CONFIG['RATE_LIMIT_PER_MINUTE']}/min",
                'formats': list(ALLOWED_MIMES)
            },
            'POST /generate-key': {
                'description': 'Gera nova API key',
                'authentication': 'API Key (admin)',
                'protected': True
            },
            'GET /health': 'Health check (público)',
            'GET /ready': 'Readiness check (público)',
            'GET /': 'Documentação (público)'
        },
        'usage_example': {
            'curl': f"curl -X POST -H '{CONFIG['API_KEY_HEADER']}: SUA-KEY-AQUI' -F 'file=@planta.pdf' https://seu-app/process --output resultado.jpg",
            'python': f"headers = {{'{CONFIG['API_KEY_HEADER']}': 'SUA-KEY-AQUI'}}\nfiles = {{'file': open('planta.pdf', 'rb')}}\nresponse = requests.post('https://seu-app/process', headers=headers, files=files)"
        },
        'environment': {
            'required': ['API_KEYS'],
            'optional': [
                'API_KEY_HEADER', 'RATE_LIMIT_PER_MINUTE',
                'RATE_LIMIT_PER_HOUR', 'RATE_LIMIT_PER_DAY',
                'ENABLE_LINE_CLEANUP', 'LINE_OVERLAP_THRESHOLD'
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

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == '__main__':
    logger.warning("⚠️  MODO DESENVOLVIMENTO")
    logger.info("🚀 Servidor: http://0.0.0.0:8000")
    
    if not CONFIG['API_KEYS']:
        logger.error("=" * 80)
        logger.error("❌ NENHUMA API KEY CONFIGURADA!")
        logger.error("🔑 Gere uma key com:")
        logger.error(f"   python -c \"import secrets; print(secrets.token_hex(32))\"")
        logger.error("")
        test_key = generate_api_key()
        logger.error(f"   Exemplo: API_KEYS={test_key}")
        logger.error("=" * 80)
    else:
        logger.info(f"✅ {len(CONFIG['API_KEYS'])} API key(s) configurada(s)")
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
