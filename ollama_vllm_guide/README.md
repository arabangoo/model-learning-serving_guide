# Ollama와 vLLM 가이드

## 목차
- [개요](#개요)
- [Ollama](#ollama)
  - [Ollama란?](#ollama란)
  - [설치](#ollama-설치)
  - [기본 사용법](#ollama-기본-사용법)
  - [모델 관리](#모델-관리)
  - [API 사용](#ollama-api-사용)
  - [커스텀 모델 생성](#커스텀-모델-생성)
- [vLLM](#vllm)
  - [vLLM이란?](#vllm이란)
  - [설치](#vllm-설치)
  - [기본 사용법](#vllm-기본-사용법)
  - [OpenAI 호환 API 서버](#openai-호환-api-서버)
  - [성능 최적화](#성능-최적화)
- [비교 및 선택 가이드](#비교-및-선택-가이드)
- [활용 사례](#활용-사례)

---

## 개요

이 가이드는 로컬 환경에서 대규모 언어 모델(LLM)을 실행하기 위한 두 가지 주요 도구인 **Ollama**와 **vLLM**에 대한 상세한 설명을 제공합니다.

- **Ollama**: 사용자 친화적인 로컬 LLM 실행 도구
- **vLLM**: 고성능 LLM 추론 및 서빙 엔진

---

## Ollama

### Ollama란?

Ollama는 로컬 환경에서 대규모 언어 모델을 쉽게 실행할 수 있도록 설계된 오픈소스 도구입니다. Docker와 유사한 방식으로 모델을 관리하며, 간단한 명령어로 다양한 LLM을 다운로드하고 실행할 수 있습니다.

**주요 특징:**
- 🚀 간편한 설치 및 사용
- 📦 다양한 사전 학습 모델 지원 (Llama 3, Mistral, Gemma 등)
- 🔧 커스텀 모델 생성 가능
- 🌐 REST API 제공
- 💻 CPU 및 GPU 모두 지원

### Ollama 설치

#### Windows

```bash
# Ollama 다운로드 및 설치
# https://ollama.ai/download 에서 설치 프로그램 다운로드
```

#### macOS

```bash
# Homebrew를 사용한 설치
brew install ollama

# 또는 공식 웹사이트에서 다운로드
# https://ollama.ai/download
```

#### Linux

```bash
# 설치 스크립트 실행
curl -fsSL https://ollama.ai/install.sh | sh

# 서비스 시작
sudo systemctl start ollama
sudo systemctl enable ollama
```

#### 설치 확인

```bash
ollama --version
```

### Ollama 기본 사용법

#### 1. 모델 실행

```bash
# Llama 3 모델 실행 (자동 다운로드)
ollama run llama3

# 대화형 모드 시작
>>> Hello, how are you?
```

#### 2. 일회성 프롬프트 실행

```bash
# 한 번만 질문하고 종료
ollama run llama3 "Explain quantum computing in simple terms"
```

#### 3. 다른 모델 사용

```bash
# Mistral 모델
ollama run mistral

# Gemma 모델
ollama run gemma:7b

# CodeLlama (코드 생성에 특화)
ollama run codellama
```

### 모델 관리

#### 모델 목록 확인

```bash
# 다운로드된 모델 목록
ollama list
```

#### 모델 다운로드

```bash
# 모델만 다운로드 (실행하지 않음)
ollama pull llama3

# 특정 크기의 모델
ollama pull llama3:70b
```

#### 모델 삭제

```bash
# 모델 제거
ollama rm llama3
```

#### 사용 가능한 모델 검색

```bash
# 모델 라이브러리 확인
# https://ollama.ai/library 방문
```

**인기 모델:**
- `llama3` - Meta의 Llama 3 모델
- `mistral` - Mistral AI의 고성능 모델
- `gemma` - Google의 경량 모델
- `codellama` - 코드 생성 특화
- `phi` - Microsoft의 소형 모델
- `neural-chat` - 대화 최적화 모델

### Ollama API 사용

Ollama는 REST API를 제공하여 프로그래밍 방식으로 모델과 상호작용할 수 있습니다.

#### Python 예제

```python
import requests
import json

# 기본 생성 요청
def generate_text(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)
    return response.json()["response"]

# 사용 예
result = generate_text("Python에서 리스트 컴프리헨션을 설명해주세요")
print(result)
```

#### 스트리밍 응답

```python
import requests
import json

def generate_streaming(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    response = requests.post(url, json=data, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            if not chunk.get("done"):
                print(chunk.get("response"), end="", flush=True)

generate_streaming("인공지능의 미래에 대해 설명해주세요")
```

#### 채팅 API

```python
def chat(messages, model="llama3"):
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    response = requests.post(url, json=data)
    return response.json()["message"]["content"]

# 대화형 사용
messages = [
    {"role": "user", "content": "Python이란 무엇인가요?"},
]

response = chat(messages)
print(response)

# 대화 이어가기
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "주요 특징을 알려주세요"})

response = chat(messages)
print(response)
```

#### JavaScript/Node.js 예제

```javascript
const axios = require('axios');

async function generateText(prompt, model = 'llama3') {
    const url = 'http://localhost:11434/api/generate';
    const response = await axios.post(url, {
        model: model,
        prompt: prompt,
        stream: false
    });

    return response.data.response;
}

// 사용
generateText('JavaScript의 async/await를 설명해주세요')
    .then(result => console.log(result));
```

### 커스텀 모델 생성

Ollama는 `Modelfile`을 사용하여 커스텀 모델을 생성할 수 있습니다.

#### Modelfile 생성

```dockerfile
# Modelfile
FROM llama3

# 시스템 프롬프트 설정
SYSTEM """
당신은 한국어를 유창하게 구사하는 AI 어시스턴트입니다.
항상 정중하고 친절하게 답변하며, 전문적인 지식을 제공합니다.
"""

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# 반복 패널티
PARAMETER repeat_penalty 1.1
```

#### 모델 생성 및 실행

```bash
# Modelfile로부터 모델 생성
ollama create my-korean-assistant -f ./Modelfile

# 커스텀 모델 실행
ollama run my-korean-assistant
```

#### 고급 Modelfile 예제 (코드 어시스턴트)

```dockerfile
FROM codellama

SYSTEM """
You are an expert software engineer specializing in Python, JavaScript, and Go.
Provide clean, efficient, and well-documented code.
Always include error handling and follow best practices.
"""

PARAMETER temperature 0.3
PARAMETER num_ctx 4096

# 템플릿 정의
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ end }}
"""
```

### Ollama 설정 및 최적화

#### 환경 변수

```bash
# GPU 메모리 제한 (GB)
export OLLAMA_GPU_MEMORY=8

# 컨텍스트 윈도우 크기
export OLLAMA_NUM_PARALLEL=4

# 서버 호스트 및 포트
export OLLAMA_HOST=0.0.0.0:11434
```

#### 모델 실행 시 파라미터 조정

```bash
ollama run llama3 --verbose \
  --temperature 0.8 \
  --top-p 0.9 \
  --repeat-penalty 1.2
```

---

## vLLM

### vLLM이란?

vLLM(Very Large Language Model)은 UC Berkeley의 LMSYS 연구실에서 개발한 고성능 LLM 추론 및 서빙 라이브러리입니다. **PagedAttention** 알고리즘을 사용하여 메모리 효율성을 극대화하고, 처리량을 크게 향상시킵니다.

**주요 특징:**
- ⚡ 최대 24배 빠른 처리량
- 💾 PagedAttention으로 메모리 효율성 향상
- 🔄 연속 배칭으로 여러 요청 동시 처리
- 🌐 OpenAI 호환 API 서버
- 🎯 텐서 병렬화 지원 (다중 GPU)
- 📊 CUDA, ROCm 지원

### vLLM 설치

#### 요구사항

- Python 3.8 이상
- CUDA 11.8 이상 (GPU 사용 시)
- Linux (권장) 또는 WSL2 (Windows)

#### pip를 통한 설치

```bash
# CUDA 12.1
pip install vllm

# 또는 특정 CUDA 버전
pip install vllm-cuda11  # CUDA 11.x용
```

#### 소스로부터 설치

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

#### Docker 사용

```bash
# vLLM Docker 이미지 실행
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model facebook/opt-125m
```

### vLLM 기본 사용법

#### Python에서 직접 사용

```python
from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(model="meta-llama/Llama-3-8b-hf")

# 샘플링 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 프롬프트
prompts = [
    "Python에서 데코레이터란 무엇인가요?",
    "머신러닝과 딥러닝의 차이는?",
]

# 생성
outputs = llm.generate(prompts, sampling_params)

# 결과 출력
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
```

#### 배치 처리

```python
from vllm import LLM, SamplingParams

# 모델 초기화
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    tensor_parallel_size=2,  # 2개 GPU 사용
    gpu_memory_utilization=0.9
)

# 대량의 프롬프트 처리
prompts = [f"Tell me about topic {i}" for i in range(100)]

sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

# 효율적인 배치 처리
outputs = llm.generate(prompts, sampling_params)
```

### OpenAI 호환 API 서버

vLLM은 OpenAI API와 호환되는 서버를 제공하여 기존 OpenAI 클라이언트 코드를 그대로 사용할 수 있습니다.

#### 서버 시작

```bash
# 기본 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-hf \
    --port 8000

# GPU 최적화 설정
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-hf \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096
```

#### OpenAI Python 클라이언트 사용

```python
from openai import OpenAI

# vLLM 서버에 연결
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM은 API 키가 필요 없음
)

# 채팅 완성
response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-hf",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user", "content": "Python의 장점을 설명해주세요"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

#### 스트리밍 응답

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# 스트리밍 모드
stream = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-hf",
    messages=[
        {"role": "user", "content": "인공지능의 역사를 설명해주세요"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### cURL을 사용한 API 호출

```bash
# 채팅 완성
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3-8b-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }'

# 텍스트 완성
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3-8b-hf",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0.8
    }'
```

### 성능 최적화

#### 1. GPU 메모리 활용 최적화

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8b-hf",
    gpu_memory_utilization=0.95,  # GPU 메모리 95% 사용
    max_model_len=4096,  # 최대 시퀀스 길이
    enforce_eager=False,  # CUDA 그래프 사용
)
```

#### 2. 다중 GPU 텐서 병렬화

```python
from vllm import LLM

# 4개 GPU에 모델 분산
llm = LLM(
    model="meta-llama/Llama-3-70b-hf",
    tensor_parallel_size=4,
    dtype="float16"
)
```

#### 3. 양자화 사용

```python
from vllm import LLM

# AWQ 양자화 모델 사용
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
    dtype="half"
)

# GPTQ 양자화
llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq"
)
```

#### 4. 프리픽스 캐싱

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8b-hf",
    enable_prefix_caching=True  # 공통 프리픽스 캐싱
)

# 시스템 프롬프트를 공유하는 여러 요청에 유리
system_prompt = "You are an expert Python programmer."
prompts = [
    system_prompt + "\n\nExplain decorators.",
    system_prompt + "\n\nExplain generators.",
    system_prompt + "\n\nExplain context managers."
]

outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

#### 5. 서버 성능 튜닝

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --disable-log-requests \
    --dtype float16
```

### 고급 기능

#### LoRA 어댑터 사용

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 기본 모델 로드
llm = LLM(
    model="meta-llama/Llama-3-8b-hf",
    enable_lora=True,
    max_lora_rank=64
)

# LoRA 어댑터와 함께 생성
outputs = llm.generate(
    "Translate to French: Hello, how are you?",
    SamplingParams(temperature=0.7, max_tokens=50),
    lora_request=LoRARequest("translation", 1, "/path/to/lora/adapter")
)
```

#### 멀티모달 모델 (비전)

```python
from vllm import LLM, SamplingParams

# LLaVA와 같은 비전-언어 모델
llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# 이미지와 텍스트 프롬프트
outputs = llm.generate({
    "prompt": "USER: <image>\nWhat is in this image?\nASSISTANT:",
    "multi_modal_data": {"image": "/path/to/image.jpg"}
})
```

---

## 비교 및 선택 가이드

### Ollama vs vLLM 비교

| 특징 | Ollama | vLLM |
|------|--------|------|
| **사용 편의성** | ⭐⭐⭐⭐⭐ 매우 쉬움 | ⭐⭐⭐ 중간 |
| **성능** | ⭐⭐⭐ 좋음 | ⭐⭐⭐⭐⭐ 뛰어남 |
| **메모리 효율성** | ⭐⭐⭐ 좋음 | ⭐⭐⭐⭐⭐ 매우 높음 |
| **처리량** | ⭐⭐⭐ 단일 요청에 적합 | ⭐⭐⭐⭐⭐ 대량 요청 처리 |
| **모델 관리** | ⭐⭐⭐⭐⭐ 매우 쉬움 | ⭐⭐⭐ 수동 관리 |
| **커스터마이징** | ⭐⭐⭐⭐ Modelfile 지원 | ⭐⭐⭐⭐⭐ 고급 옵션 풍부 |
| **다중 GPU 지원** | ⭐⭐ 제한적 | ⭐⭐⭐⭐⭐ 텐서 병렬화 |
| **API 호환성** | REST API | OpenAI 호환 API |
| **플랫폼 지원** | Windows, macOS, Linux | Linux (권장), WSL2 |
| **학습 곡선** | 낮음 | 중간-높음 |

### 언제 Ollama를 사용할까?

✅ **Ollama를 선택하세요:**
- 개인 용도 또는 소규모 프로젝트
- 빠르고 쉬운 설정이 필요할 때
- 명령줄에서 간단하게 모델을 테스트하고 싶을 때
- Docker 스타일의 모델 관리를 선호할 때
- macOS 또는 Windows 환경
- 동시 요청이 많지 않을 때

### 언제 vLLM을 사용할까?

✅ **vLLM을 선택하세요:**
- 프로덕션 환경에서 높은 처리량이 필요할 때
- 여러 GPU를 활용한 대형 모델 실행
- 동시에 많은 요청을 처리해야 할 때
- 메모리 효율성이 중요할 때
- OpenAI API 호환성이 필요할 때
- 최대 성능이 필요한 서비스 구축
- 배치 추론 작업

### 하이브리드 접근

두 도구를 함께 사용할 수도 있습니다:
- **개발**: Ollama로 빠르게 프로토타입 개발 및 테스트
- **프로덕션**: vLLM으로 고성능 서비스 배포

---

## NVIDIA GPU 서버 프로덕션 배포

### GPU 서버 환경 설정

#### 1. NVIDIA 드라이버 및 CUDA 설치

```bash
# NVIDIA 드라이버 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# NVIDIA 드라이버 설치 (Ubuntu/Debian)
sudo apt update
sudo apt install -y nvidia-driver-535

# CUDA Toolkit 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 2. Docker 및 NVIDIA Container Toolkit 설치

```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# NVIDIA Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Docker 재시작
sudo systemctl restart docker

# GPU 접근 테스트
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Ollama GPU 서버 구성

#### 1. 시스템 서비스로 Ollama 설정

```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# GPU 메모리 설정 (환경 변수)
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_GPU_OVERHEAD=0"
EOF

# 서비스 재시작
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl enable ollama

# 상태 확인
sudo systemctl status ollama
```

#### 2. Ollama Docker 배포

```bash
# docker-compose.yml 생성
cat > docker-compose.yml <<EOF
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - OLLAMA_ORIGINS=*
      - OLLAMA_NUM_PARALLEL=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama-data:
EOF

# 컨테이너 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f ollama

# 모델 다운로드
docker exec -it ollama-server ollama pull llama3
docker exec -it ollama-server ollama pull mistral
```

#### 3. Nginx 리버스 프록시 설정

```bash
# Nginx 설치
sudo apt install -y nginx

# Ollama 프록시 설정
sudo tee /etc/nginx/sites-available/ollama <<EOF
upstream ollama_backend {
    server localhost:11434;
    keepalive 32;
}

server {
    listen 80;
    server_name ollama.yourdomain.com;

    # SSL 설정 (Let's Encrypt 사용 시)
    # listen 443 ssl http2;
    # ssl_certificate /etc/letsencrypt/live/ollama.yourdomain.com/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/ollama.yourdomain.com/privkey.pem;

    client_max_body_size 100M;

    location / {
        proxy_pass http://ollama_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # 타임아웃 설정 (긴 응답 대비)
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF

# 심볼릭 링크 생성
sudo ln -s /etc/nginx/sites-available/ollama /etc/nginx/sites-enabled/

# Nginx 재시작
sudo nginx -t
sudo systemctl restart nginx
```

### vLLM GPU 서버 구성

#### 1. Python 환경 설정

```bash
# Python 3.10+ 설치
sudo apt install -y python3.10 python3.10-venv python3-pip

# 가상 환경 생성
python3.10 -m venv vllm-env
source vllm-env/bin/activate

# vLLM 설치
pip install vllm
pip install ray  # 분산 처리용
```

#### 2. vLLM 서비스 스크립트 작성

```bash
# vLLM 서비스 스크립트
cat > /opt/vllm/start_vllm.sh <<'EOF'
#!/bin/bash

MODEL_NAME="meta-llama/Llama-3-8b-hf"
TENSOR_PARALLEL_SIZE=1  # GPU 수에 맞게 조정
PORT=8000

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --dtype auto \
    --trust-remote-code
EOF

chmod +x /opt/vllm/start_vllm.sh
```

#### 3. Systemd 서비스 설정

```bash
# vLLM systemd 서비스
sudo tee /etc/systemd/system/vllm.service <<EOF
[Unit]
Description=vLLM OpenAI Compatible API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/vllm
Environment="PATH=/home/ubuntu/vllm-env/bin:/usr/local/cuda/bin:\$PATH"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/vllm/start_vllm.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl start vllm
sudo systemctl enable vllm

# 상태 확인
sudo systemctl status vllm
journalctl -u vllm -f
```

#### 4. vLLM Docker 배포 (멀티 GPU)

```yaml
# docker-compose-vllm.yml
version: '3.8'

services:
  vllm-api:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}  # 필요 시
    command: >
      --model meta-llama/Llama-3-8b-hf
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.95
      --max-model-len 4096
      --dtype auto
    shm_size: '8gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2  # 사용할 GPU 수
              capabilities: [gpu]
```

```bash
# 실행
docker-compose -f docker-compose-vllm.yml up -d

# 로그 확인
docker logs -f vllm-server
```

#### 5. Ray를 사용한 분산 vLLM 클러스터

```python
# distributed_vllm.py
from vllm import LLM, SamplingParams
import ray

# Ray 초기화
ray.init()

# 여러 노드에 걸쳐 모델 배포
@ray.remote(num_gpus=1)
class LLMWorker:
    def __init__(self, model_name):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95
        )

    def generate(self, prompts, params):
        return self.llm.generate(prompts, params)

# 여러 워커 생성
workers = [
    LLMWorker.remote("meta-llama/Llama-3-8b-hf")
    for _ in range(4)  # 4개 GPU
]

# 부하 분산
def distributed_generate(prompts, params):
    chunk_size = len(prompts) // len(workers)
    chunks = [
        prompts[i:i+chunk_size]
        for i in range(0, len(prompts), chunk_size)
    ]

    futures = [
        worker.generate.remote(chunk, params)
        for worker, chunk in zip(workers, chunks)
    ]

    results = ray.get(futures)
    return [item for sublist in results for item in sublist]
```

### 프로덕션 애플리케이션 구축

#### 1. FastAPI 기반 LLM 서비스

```python
# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import asyncio
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 백엔드 설정
OLLAMA_URL = "http://localhost:11434"
VLLM_URL = "http://localhost:8000/v1"

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 500
    backend: str = "ollama"  # "ollama" or "vllm"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 500
    stream: bool = False
    backend: str = "vllm"

# Ollama 생성
async def generate_ollama(prompt: str, model: str, temperature: float, max_tokens: int):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "options": {
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# vLLM 생성
async def generate_vllm(prompt: str, model: str, temperature: float, max_tokens: int):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{VLLM_URL}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"vLLM error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# vLLM 채팅
async def chat_vllm(messages: List[dict], model: str, temperature: float, max_tokens: int):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{VLLM_URL}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"vLLM chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "LLM Service API",
        "version": "1.0.0",
        "backends": ["ollama", "vllm"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """텍스트 생성 엔드포인트"""
    logger.info(f"Generate request - Backend: {request.backend}, Model: {request.model}")

    if request.backend == "ollama":
        result = await generate_ollama(
            request.prompt,
            request.model,
            request.temperature,
            request.max_tokens
        )
        return {
            "text": result.get("response", ""),
            "model": request.model,
            "backend": "ollama"
        }
    elif request.backend == "vllm":
        result = await generate_vllm(
            request.prompt,
            request.model,
            request.temperature,
            request.max_tokens
        )
        return {
            "text": result["choices"][0]["text"],
            "model": request.model,
            "backend": "vllm"
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid backend")

@app.post("/chat")
async def chat(request: ChatRequest):
    """채팅 엔드포인트"""
    logger.info(f"Chat request - Backend: {request.backend}, Model: {request.model}")

    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    result = await chat_vllm(
        messages,
        request.model,
        request.temperature,
        request.max_tokens
    )

    return {
        "message": result["choices"][0]["message"]["content"],
        "model": request.model,
        "backend": "vllm"
    }

@app.post("/batch")
async def batch_generate(prompts: List[str], model: str = "llama3", backend: str = "vllm"):
    """배치 처리 엔드포인트"""
    logger.info(f"Batch request - Count: {len(prompts)}, Backend: {backend}")

    tasks = [
        generate_vllm(prompt, model, 0.7, 500)
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        "results": [
            r["choices"][0]["text"] if not isinstance(r, Exception) else str(r)
            for r in results
        ],
        "count": len(prompts)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=4)
```

#### 2. 애플리케이션 배포 스크립트

```bash
# deploy.sh
#!/bin/bash

# 의존성 설치
pip install fastapi uvicorn[standard] httpx pydantic

# 서비스 파일 생성
sudo tee /etc/systemd/system/llm-api.service <<EOF
[Unit]
Description=LLM API Service
After=network.target vllm.service ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/llm-service
Environment="PATH=/home/ubuntu/vllm-env/bin:\$PATH"
ExecStart=/home/ubuntu/vllm-env/bin/uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 시작
sudo systemctl daemon-reload
sudo systemctl start llm-api
sudo systemctl enable llm-api
```

#### 3. Docker Compose 전체 스택

```yaml
# docker-compose-full.yml
version: '3.8'

services:
  # vLLM 서버
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model meta-llama/Llama-3-8b-hf
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
    shm_size: '8gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  # Ollama 서버
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  # FastAPI 애플리케이션
  llm-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: llm-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    depends_on:
      - vllm
      - ollama
    environment:
      - OLLAMA_URL=http://ollama:11434
      - VLLM_URL=http://vllm:8000/v1

  # Nginx 리버스 프록시
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - llm-api

  # Redis (캐싱용)
  redis:
    image: redis:alpine
    container_name: redis-cache
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  # Prometheus (모니터링)
  prometheus:
    image: prom/prometheus
    container_name: prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  # Grafana (대시보드)
  grafana:
    image: grafana/grafana
    container_name: grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  ollama-data:
  redis-data:
  prometheus-data:
  grafana-data:
```

#### 4. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY app.py .

# 포트 노출
EXPOSE 8080

# 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

```txt
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
httpx==0.26.0
pydantic==2.5.0
redis==5.0.1
prometheus-client==0.19.0
```

### 모니터링 및 관리

#### 1. GPU 모니터링 스크립트

```python
# gpu_monitor.py
import subprocess
import time
from prometheus_client import start_http_server, Gauge
import re

# Prometheus 메트릭 정의
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('gpu_memory_used_mb', 'GPU memory used', ['gpu_id'])
gpu_memory_total = Gauge('gpu_memory_total_mb', 'GPU memory total', ['gpu_id'])
gpu_temperature = Gauge('gpu_temperature_celsius', 'GPU temperature', ['gpu_id'])

def get_gpu_stats():
    """nvidia-smi로 GPU 통계 수집"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
         '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )

    for line in result.stdout.strip().split('\n'):
        gpu_id, util, mem_used, mem_total, temp = line.split(', ')

        gpu_utilization.labels(gpu_id=gpu_id).set(float(util))
        gpu_memory_used.labels(gpu_id=gpu_id).set(float(mem_used))
        gpu_memory_total.labels(gpu_id=gpu_id).set(float(mem_total))
        gpu_temperature.labels(gpu_id=gpu_id).set(float(temp))

if __name__ == '__main__':
    # Prometheus 서버 시작 (포트 8001)
    start_http_server(8001)

    # 5초마다 GPU 통계 수집
    while True:
        get_gpu_stats()
        time.sleep(5)
```

#### 2. Prometheus 설정

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gpu_metrics'
    static_configs:
      - targets: ['localhost:8001']

  - job_name: 'llm_api'
    static_configs:
      - targets: ['localhost:8080']

  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
```

#### 3. 로그 수집 및 분석

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging(log_file='llm_service.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 파일 핸들러 (100MB 로테이션)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100*1024*1024,
        backupCount=10
    )
    file_handler.setFormatter(JSONFormatter())

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
```

#### 4. 부하 테스트

```python
# load_test.py
import asyncio
import httpx
import time
from statistics import mean, stdev

async def make_request(client, prompt):
    start = time.time()
    try:
        response = await client.post(
            "http://localhost:8080/generate",
            json={
                "prompt": prompt,
                "model": "llama3",
                "backend": "vllm"
            },
            timeout=60.0
        )
        latency = time.time() - start
        return {"success": True, "latency": latency, "status": response.status_code}
    except Exception as e:
        return {"success": False, "latency": time.time() - start, "error": str(e)}

async def load_test(num_requests=100, concurrency=10):
    prompts = [f"Explain concept number {i}" for i in range(num_requests)]

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(0, num_requests, concurrency):
            batch = prompts[i:i+concurrency]
            batch_tasks = [make_request(client, p) for p in batch]
            tasks.extend(batch_tasks)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    # 통계 계산
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency"] for r in successful]

    print(f"\n=== Load Test Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/sec: {num_requests/total_time:.2f}")

    if latencies:
        print(f"\nLatency Statistics:")
        print(f"  Mean: {mean(latencies):.2f}s")
        print(f"  Std Dev: {stdev(latencies):.2f}s")
        print(f"  Min: {min(latencies):.2f}s")
        print(f"  Max: {max(latencies):.2f}s")

if __name__ == "__main__":
    asyncio.run(load_test(num_requests=100, concurrency=10))
```

### 보안 및 인증

#### 1. API 키 인증

```python
# auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 환경 변수나 데이터베이스에서 로드
VALID_API_KEYS = {
    "your-secret-key-1": "user1",
    "your-secret-key-2": "user2"
}

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing"
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

    return VALID_API_KEYS[api_key]

# app.py에서 사용
from fastapi import Depends

@app.post("/generate")
async def generate(
    request: GenerateRequest,
    user: str = Depends(get_api_key)
):
    # API 키 검증 후 처리
    ...
```

#### 2. Rate Limiting

```python
# rate_limiter.py
from fastapi import HTTPException
import redis
from datetime import datetime, timedelta

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def check_rate_limit(api_key: str, limit: int = 100, window: int = 3600):
    """시간당 요청 수 제한"""
    key = f"rate_limit:{api_key}"
    current = redis_client.get(key)

    if current is None:
        redis_client.setex(key, window, 1)
        return

    current = int(current)
    if current >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {limit} requests per hour."
        )

    redis_client.incr(key)
```

### 스케일링 전략

#### 1. 수평 스케일링 (Kubernetes)

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-3-8b-hf
          - --tensor-parallel-size
          - "1"
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

#### 2. 로드 밸런싱 (HAProxy)

```conf
# haproxy.cfg
global
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend vllm_front
    bind *:8000
    default_backend vllm_back

backend vllm_back
    balance roundrobin
    server vllm1 192.168.1.101:8000 check
    server vllm2 192.168.1.102:8000 check
    server vllm3 192.168.1.103:8000 check
```

## 활용 사례

### 1. 로컬 코드 어시스턴트 (Ollama)

```bash
# CodeLlama 실행
ollama run codellama

# 코드 리뷰 요청
>>> Review this Python function:
>>> def factorial(n):
>>>     return 1 if n == 0 else n * factorial(n-1)
```

### 2. 채팅봇 서비스 (vLLM)

```python
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

@app.post("/chat")
async def chat(message: str):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-hf",
        messages=[{"role": "user", "content": message}]
    )
    return {"response": response.choices[0].message.content}
```

### 3. 문서 요약 파이프라인 (Ollama)

```python
import requests

def summarize_document(text, model="llama3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": f"다음 문서를 3문장으로 요약해주세요:\n\n{text}",
            "stream": False
        }
    )
    return response.json()["response"]

# 사용
document = "긴 문서 내용..."
summary = summarize_document(document)
print(summary)
```

### 4. 대규모 데이터 분석 (vLLM)

```python
from vllm import LLM, SamplingParams
import pandas as pd

# 모델 초기화
llm = LLM(model="meta-llama/Llama-3-8b-hf")

# 대량의 고객 리뷰 분석
reviews = pd.read_csv("reviews.csv")["review_text"].tolist()

prompts = [
    f"Analyze sentiment (positive/negative/neutral): {review}"
    for review in reviews
]

# 배치 처리
results = llm.generate(
    prompts,
    SamplingParams(temperature=0.3, max_tokens=10)
)

# 결과 저장
sentiments = [output.outputs[0].text.strip() for output in results]
reviews_df["sentiment"] = sentiments
```

### 5. RAG (Retrieval-Augmented Generation) 시스템

```python
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 임베딩 모델
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 지식 베이스
documents = [
    "Python은 1991년에 발표된 프로그래밍 언어입니다.",
    "딥러닝은 인공 신경망을 사용하는 머신러닝 기법입니다.",
    # ... 더 많은 문서
]

# 벡터 데이터베이스 구축
embeddings = embedder.encode(documents)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# LLM 초기화
llm = LLM(model="meta-llama/Llama-3-8b-hf")

def rag_query(question, k=3):
    # 관련 문서 검색
    q_embedding = embedder.encode([question])
    distances, indices = index.search(q_embedding, k)

    context = "\n".join([documents[i] for i in indices[0]])

    # LLM으로 답변 생성
    prompt = f"""다음 정보를 바탕으로 질문에 답변하세요:

Context:
{context}

Question: {question}

Answer:"""

    output = llm.generate(prompt, SamplingParams(temperature=0.7, max_tokens=200))
    return output[0].outputs[0].text

# 사용
answer = rag_query("Python은 언제 만들어졌나요?")
print(answer)
```

### 6. 다국어 번역 서비스 (Ollama)

```python
import requests

class TranslationService:
    def __init__(self, model="llama3"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    def translate(self, text, source_lang, target_lang):
        prompt = f"""Translate the following text from {source_lang} to {target_lang}.
Only provide the translation, no explanations.

Text: {text}

Translation:"""

        response = requests.post(self.api_url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })

        return response.json()["response"].strip()

# 사용
translator = TranslationService()
result = translator.translate(
    "Hello, how are you?",
    "English",
    "Korean"
)
print(result)  # 안녕하세요, 어떻게 지내세요?
```

---

## 추가 리소스

### Ollama
- 공식 웹사이트: https://ollama.com
- GitHub: https://github.com/ollama/ollama
- 모델 라이브러리: https://ollama.ai/library

### vLLM
- 공식 문서: https://docs.vllm.ai
- GitHub: https://github.com/vllm-project/vllm
- 논문: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 모델 허브
- Hugging Face: https://huggingface.co/models
- Ollama Library: https://ollama.com/library

