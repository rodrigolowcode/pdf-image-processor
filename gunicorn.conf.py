"""
Gunicorn Configuration - Production Ready
Configurável via variáveis de ambiente do Easypanel
"""

import os
import multiprocessing

# ============================================================================
# CONFIGURAÇÕES VIA ENVIRONMENT VARIABLES
# ============================================================================

# Bind
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Workers (padrão: 2x CPU cores)
workers_env = os.getenv("GUNICORN_WORKERS")
if workers_env:
    workers = int(workers_env)
else:
    # Fórmula: 2 * CPU + 1 (limitado a 4 para evitar OOM)
    workers = min(multiprocessing.cpu_count() * 2 + 1, 4)

# Worker class
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "sync")

# Threads por worker (apenas para threaded/gthread)
threads = int(os.getenv("GUNICORN_THREADS", "1"))

# Timeout (importante para processamento de imagens)
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Graceful timeout
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))

# Keep alive
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# Max requests antes de reiniciar worker (previne memory leaks)
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "100"))

# Preload app (carrega antes de forkar workers)
preload_app = os.getenv("GUNICORN_PRELOAD_APP", "true").lower() == "true"

# ============================================================================
# LOGGING
# ============================================================================

# Access log
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")  # stdout

# Error log
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")  # stdout

# Log level
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# ============================================================================
# PROCESS NAMING
# ============================================================================

proc_name = os.getenv("GUNICORN_PROC_NAME", "pdf-image-processor")

# ============================================================================
# SERVER MECHANICS
# ============================================================================

# Backlog de conexões
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# Worker temporary directory
worker_tmp_dir = os.getenv("GUNICORN_WORKER_TMP_DIR", "/dev/shm")

# ============================================================================
# HOOKS
# ============================================================================

def on_starting(server):
    """Executado quando Gunicorn inicia"""
    server.log.info("=" * 80)
    server.log.info("🚀 PDF/Image Processor v3.1 iniciando")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Timeout: {timeout}s")
    server.log.info(f"Max requests: {max_requests}")
    server.log.info(f"Preload app: {preload_app}")
    server.log.info("=" * 80)

def when_ready(server):
    """Executado quando servidor está pronto"""
    server.log.info("✅ Servidor pronto para receber requisições")

def worker_int(worker):
    """Executado quando worker recebe SIGINT"""
    worker.log.info(f"⚠️  Worker {worker.pid} recebeu SIGINT")

def worker_abort(worker):
    """Executado quando worker é abortado (timeout)"""
    worker.log.error(f"❌ Worker {worker.pid} abortado (timeout ou crash)")

def pre_fork(server, worker):
    """Executado antes de criar worker"""
    server.log.info(f"🔄 Forking worker...")

def post_fork(server, worker):
    """Executado após criar worker"""
    server.log.info(f"✅ Worker {worker.pid} iniciado")

def pre_exec(server):
    """Executado antes de executar novo master process"""
    server.log.info("🔄 Executando novo master process")

def on_exit(server):
    """Executado quando Gunicorn sai"""
    server.log.info("👋 Servidor finalizando")

def child_exit(server, worker):
    """Executado quando worker sai"""
    server.log.info(f"👋 Worker {worker.pid} finalizado")

# ============================================================================
# INFORMAÇÕES DE DEBUG
# ============================================================================

print(f"""
╔══════════════════════════════════════════════════════════════╗
║         PDF/Image Processor - Gunicorn Configuration         ║
╚══════════════════════════════════════════════════════════════╝

📊 Configuração:
   • Workers: {workers}
   • Threads/worker: {threads}
   • Worker class: {worker_class}
   • Timeout: {timeout}s
   • Max requests: {max_requests}
   • Preload: {preload_app}

🌐 Network:
   • Bind: {bind}
   • Backlog: {backlog}
   • Keepalive: {keepalive}s

📝 Logging:
   • Access log: {accesslog}
   • Error log: {errorlog}
   • Level: {loglevel}

💾 Memória estimada:
   • RAM/worker: ~150MB
   • Total estimado: ~{workers * 150}MB

⚙️  Variáveis de ambiente disponíveis:
   GUNICORN_WORKERS, GUNICORN_TIMEOUT, GUNICORN_MAX_REQUESTS
   REDIS_URL (para Flask-Limiter)

""")
