# PDF/Image Processor v3.1

Serviço de processamento de PDFs e imagens otimizado para produção com Gunicorn, Redis e OpenCV.

## 🚀 Features

- ✅ Converte PDF para imagem (200 DPI)
- ✅ Melhora qualidade de imagens (upscale, CLAHE, denoising, sharpening)
- ✅ Processamento em memória (sem race conditions)
- ✅ Rate limiting com Redis
- ✅ Health checks para Easypanel/Kubernetes
- ✅ Configurável via variáveis de ambiente
- ✅ Otimizado para CPU (sem GPU)

## 📊 Performance

- **Latência média:** ~1.35s por imagem (1200px)
- **RAM por request:** 2-3x o tamanho do arquivo (~90-120MB para 30MB)
- **Throughput:** Limitado por CPU e workers configurados

## 🔧 Configuração no Easypanel

### 1. Deploy do Serviço

1. Conecte seu repositório GitHub no Easypanel
2. Crie novo serviço → App
3. Easypanel detectará automaticamente o Dockerfile
4. Configure porta: `8000`

### 2. Variáveis de Ambiente

Workers (2-4 recomendado para 2GB RAM)
GUNICORN_WORKERS=2

Redis (se disponível)
REDIS_URL=redis://redis:6379/0

Rate limiting
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_PER_HOUR=100

Timeout
GUNICORN_TIMEOUT=120

text

### 3. Redis (Opcional mas Recomendado)

**Opção A: Redis no Easypanel**
1. Adicione serviço Redis ao mesmo projeto
2. Configure `REDIS_URL=redis://redis:6379/0`

**Opção B: Redis externo**
REDIS_URL=redis://:senha@seu-redis.cloud:6379/0

text

**Opção C: Sem Redis (fallback)**
REDIS_URL=memory://

text
⚠️ Rate limiting não funciona entre workers com `memory://`

### 4. Health Checks

Configure no Easypanel:
- **Endpoint:** `/health`
- **Interval:** 30s
- **Timeout:** 10s

## 📡 API Endpoints

### POST /process

Processa PDF ou imagem.

**Request:**
curl -X POST http://localhost:8000/process
-F "file=@planta.pdf"
--output resultado.jpg


**Response:**
- Success: `200` - Imagem JPEG processada
- Error: `400` - Validação falhou
- Error: `429` - Rate limit excedido
- Error: `500` - Erro interno

### GET /health

Health check para monitoramento.

**Response:**
{
"status": "healthy",
"service": "pdf-image-processor",
"version": "3.1",
"worker_pid": 123
}


### GET /ready

Readiness check para Kubernetes.

## 🐳 Deploy Local

Build
docker build -t pdf-processor .

Run
docker run -d
-p 8000:8000
-e GUNICORN_WORKERS=2
-e REDIS_URL=memory://
--name pdf-processor
pdf-processor

Test
curl http://localhost:8000/health

text

## 🔍 Monitoramento

### Logs
Easypanel: Ver logs no dashboard
Docker: docker logs -f pdf-processor


### Métricas Importantes
- Latência P95 (deve ser < 3s)
- Taxa de erro (deve ser < 1%)
- Uso de RAM por worker (~150MB base + requests)
- Rate limit hits

## 🛠️ Troubleshooting

### Workers travando (timeout)
Aumente timeout
GUNICORN_TIMEOUT=180


### OOM (Out of Memory)
Reduza workers
GUNICORN_WORKERS=2

Ou limite tamanho de arquivo
MAX_CONTENT_LENGTH=20971520 # 20MB


### Rate limit muito rígido
RATE_LIMIT_PER_MINUTE=20
RATE_LIMIT_PER_HOUR=200


## 📈 Scaling

### Cálculo de Workers

workers = (RAM_GB * 0.7) / 0.15

Exemplo: 2GB → (2 * 0.7) / 0.15 = ~9 workers max
Mas comece com 2-4 e monitore
text

### Horizontal Scaling
- Use Redis para rate limiting compartilhado
- Configure load balancer no Easypanel
- Cada instância pode ter 2-4 workers

## 🔒 Segurança

- ✅ Validação de MIME type antes de salvar
- ✅ Limite de tamanho de arquivo
- ✅ Validação de dimensões
- ✅ Rate limiting por IP
- ✅ Usuário não-root no container
- ✅ Sem persistência de arquivos temporários

## 📝 Licença

MIT
