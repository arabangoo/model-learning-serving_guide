# RunPod 모델 배포 & 서비스 적용 매뉴얼

## 목차
- [개요](#개요)
- [RunPod란?](#runpod란)
- [계정 생성 및 설정](#계정-생성-및-설정)
- [GPU Pod vs Serverless 비교](#gpu-pod-vs-serverless-비교)
- [Serverless - vLLM으로 LLM 배포](#serverless---vllm으로-llm-배포)
  - [vLLM Worker 배포 단계별 가이드](#vllm-worker-배포-단계별-가이드)
  - [환경 변수 설정](#환경-변수-설정)
  - [배포 확인 및 테스트](#배포-확인-및-테스트)
- [API 호출 방법](#api-호출-방법)
  - [OpenAI 호환 API](#openai-호환-api)
  - [Python - openai 라이브러리](#python---openai-라이브러리)
  - [Python - requests 라이브러리](#python---requests-라이브러리)
  - [스트리밍 응답 처리](#스트리밍-응답-처리)
- [서비스 적용 예시](#서비스-적용-예시)
  - [FastAPI 서버 연동](#fastapi-서버-연동)
  - [Spring Boot 연동](#spring-boot-연동)
- [Custom Serverless Worker 구축](#custom-serverless-worker-구축)
- [GPU Pod로 대화형 개발 환경 구성](#gpu-pod로-대화형-개발-환경-구성)
- [비용 관리 및 스케일링](#비용-관리-및-스케일링)
- [운영 환경 Best Practice](#운영-환경-best-practice)
- [자주 발생하는 오류 및 해결 방법](#자주-발생하는-오류-및-해결-방법)

---

## 개요

이 매뉴얼은 RunPod 클라우드 GPU 플랫폼을 통해 대형 언어 모델(LLM)을 배포하고, 실제 서비스에서 API로 호출하는 방법을 설명합니다.

RunPod에서 vLLM 기반 서버리스 엔드포인트를 구성하면 OpenAI API와 동일한 방식으로 자체 호스팅 모델을 호출할 수 있습니다.

---

## RunPod란?

RunPod는 AI/ML 워크로드를 위한 클라우드 GPU 플랫폼이다.
Hugging Face의 다양한 오픈소스 모델을 GPU 서버에 올려 API로 서비스할 수 있다.

**주요 특징:**
- 시간 단위 과금의 GPU 서버 (Pod) 제공
- 서버리스(Serverless) 엔드포인트: 요청이 있을 때만 GPU 자원 소비
- vLLM 기반 LLM 서빙 지원 (OpenAI API 호환)
- Hugging Face 모델 직접 배포 가능
- A100, H100, RTX 4090 등 다양한 GPU 선택 가능
- 자동 스케일링 지원

**활용 시나리오:**
- 오픈소스 모델(Llama, Mistral, Qwen, DeepSeek 등)을 직접 호스팅하여 서비스
- 민감한 데이터를 외부 AI 서비스에 보내지 않고 자체 서버에서 처리
- 파인튜닝한 커스텀 모델을 API로 서비스
- 대용량 배치 추론 처리

---

## 계정 생성 및 설정

### 1. 계정 생성

(1) [https://www.runpod.io](https://www.runpod.io) 접속 후 **Sign Up** 클릭   
(2) Google 계정 또는 이메일로 회원가입   
(3) 이메일 인증 완료   

### 2. 크레딧 충전

(1) 대시보드 좌측 메뉴 → **Billing** 클릭   
(2) 신용카드 등록 후 원하는 금액 충전 (최소 $10)   
(3) 처음 $10 충전 시 $5~$500 사이의 랜덤 크레딧 보너스 지급   

> RunPod는 선불 크레딧 방식으로 동작한다. 사용한 GPU 시간만큼 크레딧이 차감된다.

### 3. API 키 발급

(1) 대시보드 우측 상단 프로필 클릭 → **Settings** 선택   
(2) **API Keys** 탭으로 이동   
(3) **+ API Key** 버튼 클릭하여 새 키 생성   
(4) 생성된 API 키를 즉시 복사하여 안전한 곳에 보관   

```
RUNPOD_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. 환경 변수 설정

```bash
# Linux / macOS
export RUNPOD_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows (PowerShell)
$env:RUNPOD_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

`.env` 파일 활용:
```
RUNPOD_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_ENDPOINT_ID=xxxxxxxxxxxxxxxx
```

---

## GPU Pod vs Serverless 비교

RunPod는 두 가지 주요 실행 방식을 제공한다.

| 항목 | GPU Pod | Serverless |
|------|---------|------------|
| 과금 방식 | 시간 단위 (항상 과금) | 요청 처리 시간만 과금 |
| 스케일링 | 수동 | 자동 스케일링 |
| 응답 지연 | 낮음 (서버 상시 실행) | 콜드 스타트 있음 |
| 적합한 용도 | 모델 개발, 지속적 트래픽 | API 서비스, 간헐적 요청 |
| 설정 복잡도 | 높음 (직접 환경 구성) | 낮음 (템플릿 제공) |
| 최소 비용 | 사용하지 않아도 과금 | 사용한 만큼만 과금 |

> **서비스 운영 목적으로는 Serverless 방식이 권장된다.**
> 개발/학습/파인튜닝은 GPU Pod 방식을 사용한다.

---

## Serverless - vLLM으로 LLM 배포

RunPod는 vLLM 기반의 서버리스 LLM 엔드포인트를 손쉽게 배포할 수 있는 템플릿을 제공한다.

### vLLM Worker 배포 단계별 가이드

#### 1단계: Serverless 메뉴 진입

(1) 대시보드 좌측 메뉴 → **Serverless** 클릭   
(2) **+ New Endpoint** 버튼 클릭   
(3) **Explore** 탭에서 **vLLM** 검색 후 선택   
(4) **Deploy** 버튼 클릭   

#### 2단계: 모델 설정

배포 화면에서 아래 항목을 설정한다.

**(1) Model Name (Hugging Face 모델 ID)**

배포하려는 모델의 Hugging Face 모델 ID를 입력한다.

```
# 예시
meta-llama/Meta-Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
Qwen/Qwen2.5-7B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

**(2) Max Model Length (컨텍스트 길이)**

모델이 처리할 최대 토큰 수를 지정한다.
GPU VRAM에 따라 설정 가능 범위가 달라진다.

```
# 권장값 예시
8192    # 일반적인 대화
16384   # 긴 문서 처리
32768   # 매우 긴 컨텍스트 (대용량 GPU 필요)
```

**(3) HuggingFace Token (Gated 모델인 경우)**

Llama 등 접근 승인이 필요한 모델은 Hugging Face 액세스 토큰이 필요하다.

- [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 에서 토큰 생성
- `Read` 권한의 토큰 생성 후 입력

#### 3단계: GPU 및 스케일링 설정

| 설정 항목 | 권장값 |
|-----------|--------|
| GPU 유형 | A40, A100, RTX 4090 (모델 크기에 따라 선택) |
| GPU VRAM | 7B 모델 → 16GB 이상, 13B 모델 → 24GB 이상, 70B 모델 → 80GB+ |
| Active Workers | 0 (비용 절감 기본값) |
| Max Workers | 3 (트래픽에 따라 조정) |
| Idle Timeout | 5분 (사용 없을 때 워커 종료까지 대기 시간) |

#### 4단계: 엔드포인트 생성

(1) **Create Endpoint** 버튼 클릭   
(2) 모델 다운로드 및 초기화 완료까지 수 분~수십 분 대기   
(3) 엔드포인트 상태가 **Ready** 로 변경되면 사용 가능   

#### 5단계: Endpoint ID 확인

엔드포인트 생성 후 상세 페이지에서 **Endpoint ID**를 확인한다.
API 호출 시 URL에 이 ID가 사용된다.

```
Endpoint ID: abc1234xyz
API URL: https://api.runpod.ai/v2/abc1234xyz/openai/v1
```

---

### 환경 변수 설정

vLLM 배포 후 환경 변수를 통해 세부 동작을 조정할 수 있다.
엔드포인트 상세 페이지 → **Edit** → **Environment Variables** 탭에서 설정한다.

| 환경 변수 | 설명 | 예시 |
|-----------|------|------|
| `MODEL_NAME` | Hugging Face 모델 ID | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `MAX_MODEL_LEN` | 최대 컨텍스트 길이 (토큰 수) | `8192` |
| `DTYPE` | 모델 정밀도 | `bfloat16` / `float16` / `float32` |
| `GPU_MEMORY_UTILIZATION` | GPU VRAM 사용 비율 | `0.9` |
| `QUANTIZATION` | 양자화 방식 | `awq` / `gptq` / `fp8` |
| `TENSOR_PARALLEL_SIZE` | 멀티 GPU 병렬 처리 수 | `2` |
| `CUSTOM_CHAT_TEMPLATE` | 커스텀 채팅 템플릿 | (Jinja2 형식 문자열) |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | API에서 사용할 모델 명칭 | `my-model` |
| `HUGGING_FACE_HUB_TOKEN` | Hugging Face 액세스 토큰 | `hf_xxxx...` |

---

### 배포 확인 및 테스트

배포가 완료되면 RunPod 대시보드에서 직접 테스트할 수 있다.

(1) 엔드포인트 상세 페이지 → **Requests** 탭 클릭   
(2) 기본 테스트 입력으로 요청 전송   

```json
{
    "input": {
        "prompt": "Hello, how are you?"
    }
}
```

(3) 정상 응답 예시   

```json
{
    "id": "req_xxxxxxxx",
    "status": "COMPLETED",
    "output": "I'm doing well, thank you for asking!",
    "executionTime": 1234
}
```

---

## API 호출 방법

### OpenAI 호환 API

RunPod vLLM 엔드포인트는 OpenAI API 형식과 완전히 호환된다.
`base_url`만 RunPod 엔드포인트 URL로 변경하면 기존 코드를 그대로 사용할 수 있다.

**API URL 형식:**
```
https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1
```

예시:
```
https://api.runpod.ai/v2/abc1234xyz/openai/v1
```

**인증:**
```
Authorization: Bearer {RUNPOD_API_KEY}
```

---

### Python - openai 라이브러리

```bash
pip install openai python-dotenv
```

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")  # 예: "abc1234xyz"

client = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
)

# 기본 채팅 요청
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",  # 배포한 모델 ID
    messages=[
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
        {"role": "user", "content": "머신러닝과 딥러닝의 차이점을 설명해주세요."}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
print(f"\n사용 토큰 - 입력: {response.usage.prompt_tokens}, 출력: {response.usage.completion_tokens}")
```

> 모델 ID는 `OPENAI_SERVED_MODEL_NAME_OVERRIDE` 환경 변수로 지정한 이름을 사용하거나,
> 배포 시 입력한 Hugging Face 모델 ID를 사용한다.

---

### Python - requests 라이브러리

```python
import requests
import json
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"

def call_runpod(prompt: str, system: str = "You are a helpful assistant.") -> str:
    response = requests.post(
        url=f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# 사용 예
result = call_runpod("파이썬 리스트와 튜플의 차이를 설명해주세요.")
print(result)
```

---

### 스트리밍 응답 처리

```python
import os
from openai import OpenAI

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

client = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
)

# 스트리밍 호출
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "인공지능의 발전 역사를 단계별로 설명해주세요."}
    ],
    stream=True,
    max_tokens=2048
)

# 청크 단위 실시간 출력
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # 줄바꿈
```

---

## 서비스 적용 예시

### FastAPI 서버 연동

```bash
pip install fastapi uvicorn openai python-dotenv
```

```python
# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

app = FastAPI()

client = OpenAI(
    api_key=RUNPOD_API_KEY,
    base_url=f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
)


class ChatRequest(BaseModel):
    message: str
    system: str = "당신은 친절한 AI 어시스턴트입니다."
    temperature: float = 0.7
    max_tokens: int = 2048


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.message}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    def generate():
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.message}
            ],
            stream=True,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                data = {"content": chunk.choices[0].delta.content}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# 실행: uvicorn main:app --reload --port 8000
```

---

### Spring Boot 연동

#### 방법 1: Spring AI (권장)

`pom.xml` 의존성 추가:
```xml
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
    <version>1.0.0</version>
</dependency>
```

`application.yml` 설정:
```yaml
runpod:
  api-key: ${RUNPOD_API_KEY}
  endpoint-id: ${RUNPOD_ENDPOINT_ID}
  model-name: meta-llama/Meta-Llama-3.1-8B-Instruct

spring:
  ai:
    openai:
      api-key: ${RUNPOD_API_KEY}
      base-url: https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/openai/v1
      chat:
        options:
          model: ${runpod.model-name}
          temperature: 0.7
```

Java 코드:
```java
@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    @PostMapping("/chat")
    public Map<String, String> chat(@RequestBody Map<String, String> request) {
        String response = chatClient.prompt()
            .system("당신은 친절한 AI 어시스턴트입니다.")
            .user(request.get("message"))
            .call()
            .content();
        return Map.of("response", response);
    }
}
```

#### 방법 2: WebClient (직접 HTTP 호출)

```java
@Service
public class RunPodService {

    private final WebClient webClient;
    private final String modelName;

    public RunPodService(
        @Value("${RUNPOD_API_KEY}") String apiKey,
        @Value("${RUNPOD_ENDPOINT_ID}") String endpointId,
        @Value("${runpod.model-name}") String modelName
    ) {
        this.modelName = modelName;
        this.webClient = WebClient.builder()
            .baseUrl("https://api.runpod.ai/v2/" + endpointId + "/openai/v1")
            .defaultHeader("Authorization", "Bearer " + apiKey)
            .defaultHeader("Content-Type", "application/json")
            .build();
    }

    public String chat(String userMessage) {
        Map<String, Object> requestBody = Map.of(
            "model", modelName,
            "messages", List.of(
                Map.of("role", "system", "content", "당신은 친절한 AI 어시스턴트입니다."),
                Map.of("role", "user", "content", userMessage)
            ),
            "temperature", 0.7,
            "max_tokens", 2048
        );

        Map<?, ?> response = webClient.post()
            .uri("/chat/completions")
            .bodyValue(requestBody)
            .retrieve()
            .bodyToMono(Map.class)
            .block();

        List<?> choices = (List<?>) response.get("choices");
        Map<?, ?> message = (Map<?, ?>) ((Map<?, ?>) choices.get(0)).get("message");
        return (String) message.get("content");
    }
}
```

---

## Custom Serverless Worker 구축

기본 vLLM 템플릿 외에 커스텀 로직이 필요한 경우 직접 Serverless Worker를 만들 수 있다.

### 1. 핸들러 함수 작성

```python
# handler.py
import runpod
from transformers import pipeline

# 모델 초기화 (워커 시작 시 1회 실행)
model = pipeline("text-generation", model="gpt2")

def handler(event):
    """
    RunPod Serverless 핸들러 함수
    event["input"] 에서 입력 데이터를 받는다
    """
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "")
    max_new_tokens = input_data.get("max_new_tokens", 200)

    if not prompt:
        return {"error": "prompt가 필요합니다."}

    result = model(prompt, max_new_tokens=max_new_tokens, do_sample=True)
    generated_text = result[0]["generated_text"]

    return {"output": generated_text}


# RunPod Serverless Worker 시작
runpod.serverless.start({"handler": handler})
```

### 2. 로컬 테스트

```bash
# 테스트 입력 파일 생성
cat > test_input.json << 'EOF'
{
    "input": {
        "prompt": "Once upon a time",
        "max_new_tokens": 100
    }
}
EOF

# 로컬 실행
python handler.py
```

### 3. Dockerfile 작성

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 핸들러 파일 복사
COPY handler.py .

# RunPod Worker 실행
CMD ["python3", "-u", "handler.py"]
```

`requirements.txt`:
```
runpod
transformers
torch
accelerate
```

### 4. Docker 이미지 빌드 및 푸시

```bash
# 이미지 빌드 (linux/amd64 플랫폼 필수)
docker build --platform linux/amd64 -t yourdockerhub/my-llm-worker:latest .

# Docker Hub에 푸시
docker push yourdockerhub/my-llm-worker:latest
```

### 5. RunPod에 배포

(1) 대시보드 → **Serverless** → **+ New Endpoint**   
(2) **Custom** 탭 선택   
(3) Container Image에 `docker.io/yourdockerhub/my-llm-worker:latest` 입력   
(4) GPU 타입 및 스케일링 설정   
(5) **Deploy Endpoint** 클릭   

### 6. Custom Worker API 호출

```python
import runpod
import os

runpod.api_key = os.environ.get("RUNPOD_API_KEY")

endpoint = runpod.Endpoint(os.environ.get("RUNPOD_ENDPOINT_ID"))

# 동기 호출 (결과 반환까지 대기)
result = endpoint.run_sync({
    "input": {
        "prompt": "Once upon a time",
        "max_new_tokens": 200
    }
})

print(result)

# 비동기 호출 (Job ID 즉시 반환)
job = endpoint.run({
    "input": {
        "prompt": "In a galaxy far away",
        "max_new_tokens": 300
    }
})

print(f"Job ID: {job.job_id}")

# 결과 조회 (나중에)
output = job.output()
print(output)
```

---

## GPU Pod로 대화형 개발 환경 구성

모델 개발, 파인튜닝, 실험 목적으로는 GPU Pod를 사용한다.

### Pod 생성

(1) 대시보드 → **Pods** → **+ Deploy** 클릭   
(2) GPU 타입 선택 (예: RTX 3090, A40, A100)   
(3) 템플릿 선택: `RunPod Pytorch 2.x` 또는 `RunPod CUDA 12.x`   
(4) 볼륨 크기 설정 (모델 저장 공간)   
(5) **Deploy** 클릭   

### Pod에서 vLLM 서버 실행

SSH 또는 Jupyter 접속 후:

```bash
# vLLM 설치
pip install vllm

# vLLM API 서버 실행
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

### Pod API 호출

Pod의 외부 노출 포트를 통해 API 호출:

```python
from openai import OpenAI

# Pod의 공개 URL로 접속
client = OpenAI(
    api_key="not-used",
    base_url="https://{POD_ID}-8000.proxy.runpod.net/v1"
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "안녕하세요!"}]
)
print(response.choices[0].message.content)
```

---

## 비용 관리 및 스케일링

### Serverless 과금 방식

- **Active Workers**: 24시간 상시 실행 (일정한 트래픽에 적합, 저단가)
- **Flex Workers**: 요청이 있을 때만 실행 (간헐적 트래픽에 적합, 더 비쌈)
- 과금 기준: GPU 초당 사용료 × 처리 시간

### 비용 절감 팁

| 방법 | 설명 |
|------|------|
| Idle Timeout 단축 | 유휴 상태 워커를 빨리 종료시켜 비용 절감 |
| 양자화 모델 사용 | AWQ/GPTQ 양자화로 더 작은 GPU에서 실행 |
| Active Workers 최소화 | 트래픽이 적으면 0으로 설정 |
| 소형 모델 선택 | 7B → 3B 모델로 GPU 비용 절감 |
| 배치 처리 활용 | 여러 요청을 묶어서 처리 |

### 스케일링 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| Min Workers | 0 | 유휴 시 워커 수 (0이면 콜드 스타트 발생) |
| Max Workers | 3~10 | 최대 동시 실행 워커 수 |
| Idle Timeout | 5~30분 | 요청 없을 때 워커 유지 시간 |

---

## 운영 환경 Best Practice

### 1. 환경 변수로 민감정보 관리

```python
# ❌ 코드에 직접 포함 금지
api_key = "your-actual-api-key"
endpoint_id = "abc1234xyz"

# ✅ 환경 변수로 관리
import os
api_key = os.environ.get("RUNPOD_API_KEY")
endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
```

### 2. 콜드 스타트 대응

서버리스 특성상 유휴 후 첫 요청은 모델 로딩 시간이 걸린다.

```python
import time
import requests
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

def call_with_timeout(prompt: str, timeout: int = 120) -> str:
    """콜드 스타트를 고려한 긴 타임아웃 설정"""
    response = requests.post(
        url=f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=timeout  # 첫 요청은 모델 로딩 시간 포함
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

### 3. 예외 처리 및 재시도

```python
import time
import requests
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

def call_with_retry(prompt: str, max_retries: int = 3) -> str:
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print(f"타임아웃 (시도 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(10)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status == 429:
                print("Rate Limit 초과, 잠시 대기...")
                time.sleep(30)
            elif status >= 500:
                print(f"서버 오류 {status}, 재시도...")
                time.sleep(5 * (attempt + 1))
            else:
                raise

    raise Exception(f"{max_retries}회 재시도 후 실패")
```

### 4. 헬스체크

```python
def health_check(endpoint_id: str, api_key: str) -> bool:
    """엔드포인트 상태 확인"""
    try:
        response = requests.get(
            url=f"https://api.runpod.ai/v2/{endpoint_id}/health",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        return response.status_code == 200
    except Exception:
        return False
```

---

## 자주 발생하는 오류 및 해결 방법

| 오류 | 원인 | 해결 방법 |
|------|------|-----------|
| `401 Unauthorized` | API 키 오류 | Settings → API Keys에서 키 재확인 |
| `404 Not Found` | 엔드포인트 ID 오류 | Serverless 메뉴에서 Endpoint ID 재확인 |
| 요청이 오랫동안 Queue 상태 | 워커 수 부족 또는 콜드 스타트 | Max Workers 증가 또는 5~10분 대기 |
| `CUDA out of memory` | GPU VRAM 부족 | GPU 크기 업그레이드 또는 양자화 모델 사용 |
| 모델 로딩 실패 | HuggingFace 토큰 없음 또는 모델 ID 오류 | `HUGGING_FACE_HUB_TOKEN` 환경 변수 확인 |
| `context_length_exceeded` | 입력이 MAX_MODEL_LEN 초과 | MAX_MODEL_LEN 증가 또는 입력 길이 축소 |
| 느린 응답 속도 | Active Workers 없음, Flex Workers만 사용 | 지속적 트래픽이라면 Active Workers 1 이상 설정 |

---

> **참고 자료**
> - RunPod 공식 문서: https://docs.runpod.io
> - vLLM Serverless 시작 가이드: https://docs.runpod.io/serverless/vllm/get-started
> - RunPod Python SDK: https://github.com/runpod/runpod-python
> - 서버리스 vs Pod 비교: https://www.runpod.io/articles/comparison/serverless-gpu-deployment-vs-pods
> - 가격 정책: https://www.runpod.io/pricing
