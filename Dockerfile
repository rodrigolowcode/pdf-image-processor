FROM python:3.12-slim-bookworm

LABEL maintainer="Strevo AI Platform"
LABEL version="3.1"
LABEL description="PDF/Image Processor optimized for production"

# Variáveis de build
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ============================================================================
# DEPENDÊNCIAS DO SISTEMA
# ============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # PDF processing
    poppler-utils \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # python-magic dependencies
    libmagic1 \
    # Timezone data
    tzdata \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# ============================================================================
# DEPENDÊNCIAS PYTHON
# ============================================================================

# Copiar requirements primeiro (cache de layers)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# ============================================================================
# CÓDIGO DA APLICAÇÃO
# ============================================================================

# Copiar código
COPY app.py .
COPY gunicorn.conf.py .

# ============================================================================
# CONFIGURAÇÃO DE USUÁRIO
# ============================================================================

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Criar diretórios de trabalho
RUN mkdir -p /tmp/uploads /tmp/processed && \
    chown -R appuser:appuser /tmp/uploads /tmp/processed

# Trocar para usuário não-root
USER appuser

# ============================================================================
# VARIÁVEIS DE AMBIENTE PADRÃO
# ============================================================================

# Timezone
ENV TZ=America/Sao_Paulo

# Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Gunicorn (pode ser sobrescrito no Easypanel)
ENV GUNICORN_WORKERS=2
ENV GUNICORN_TIMEOUT=120
ENV GUNICORN_MAX_REQUESTS=1000
ENV GUNICORN_BIND=0.0.0.0:8000

# Redis (configurar no Easypanel)
ENV REDIS_URL=memory://

# ============================================================================
# HEALTH CHECK
# ============================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# ============================================================================
# EXPOSIÇÃO DE PORTA
# ============================================================================

EXPOSE 8000

# ============================================================================
# COMANDO DE INICIALIZAÇÃO
# ============================================================================

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
