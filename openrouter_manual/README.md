# OpenRouter 모델 호출 & 서비스 적용 매뉴얼

## 목차
- [개요](#개요)
- [OpenRouter란?](#openrouter란)
- [계정 생성 및 API 키 발급](#계정-생성-및-api-키-발급)
- [크레딧 충전](#크레딧-충전)
- [사용 가능한 모델 확인](#사용-가능한-모델-확인)
- [무료 모델 활용](#무료-모델-활용)
- [API 기본 사용법](#api-기본-사용법)
  - [엔드포인트 및 인증](#엔드포인트-및-인증)
  - [Python - requests 라이브러리](#python---requests-라이브러리)
  - [Python - openai 라이브러리](#python---openai-라이브러리)
  - [스트리밍 응답 처리](#스트리밍-응답-처리)
- [지원 파라미터](#지원-파라미터)
- [서비스 적용 예시](#서비스-적용-예시)
  - [FastAPI 서버 연동](#fastapi-서버-연동)
  - [Spring Boot 연동](#spring-boot-연동)
- [멀티모달(이미지) 모델 호출](#멀티모달이미지-모델-호출)
- [비용 모니터링 및 사용량 확인](#비용-모니터링-및-사용량-확인)
- [운영 환경 Best Practice](#운영-환경-best-practice)
- [자주 발생하는 오류 및 해결 방법](#자주-발생하는-오류-및-해결-방법)

---

## 개요

이 매뉴얼은 OpenRouter를 통해 다양한 AI 언어 모델을 단일 API로 호출하고, 실제 서비스에 적용하는 방법을 설명합니다.

OpenRouter는 OpenAI API와 호환되는 통합 LLM 게이트웨이입니다.
GPT-4o, Claude 3.5, Llama 3, Gemini, Mistral 등 수백 가지 모델을 **하나의 API 키**로 사용할 수 있습니다.

---

## OpenRouter란?

OpenRouter는 여러 AI 모델 제공사(OpenAI, Anthropic, Meta, Google, Mistral 등)의 모델을 단일 엔드포인트로 통합 제공하는 API 플랫폼이다.

**주요 특징:**
- 수백 가지 이상의 LLM 모델을 하나의 API 키로 접근
- OpenAI API와 동일한 요청/응답 포맷 (쉬운 마이그레이션)
- 자동 폴백(Fallback): 특정 제공사 장애 시 다른 제공사로 자동 전환
- 무료 모델 제공 (일부 모델은 무료로 사용 가능)
- 프롬프트/응답 무기록 정책 (기본값: 로깅 없음)
- 제공사 직접 요금과 동일한 가격 + 안정성 향상

**활용 시나리오:**
- 여러 모델을 비교 테스트하며 최적 모델 선정
- 특정 제공사 의존도 없이 유연한 모델 스위칭
- 저비용으로 다양한 오픈소스 모델 서비스 적용
- 무료 모델로 프로토타입 개발 및 테스트

---

## 계정 생성 및 API 키 발급

### 1. 계정 생성

(1) [https://openrouter.ai](https://openrouter.ai) 접속 후 **Sign In** 클릭
(2) Google 계정, GitHub 계정 또는 이메일로 가입
(3) 가입 완료 후 대시보드로 이동

### 2. API 키 발급

(1) 우측 상단 프로필 아이콘 클릭 → **Keys** 메뉴 선택
(2) **Create Key** 버튼 클릭
(3) 키 이름(Name) 입력 후 생성
(4) 생성된 API 키를 복사하여 안전한 곳에 보관

> ⚠️ API 키는 생성 직후에만 전체 내용을 확인할 수 있습니다. 반드시 즉시 복사해두세요.

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 환경 변수 설정

운영 환경에서는 API 키를 코드에 직접 포함시키지 말고 환경 변수로 관리한다.

```bash
# Linux / macOS
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"

# Windows (PowerShell)
$env:OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"
```

`.env` 파일 활용 (python-dotenv):
```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

---

## 크레딧 충전

(1) 대시보드 상단 메뉴 → **Credits** 클릭
(2) 충전할 금액 입력 (최소 $5)
(3) 신용카드 정보 입력 후 결제
(4) 충전된 크레딧은 API 호출 시 토큰 단위로 차감됨

> 무료 모델을 사용할 경우 크레딧 없이도 사용 가능하나, 일일 요청 수에 제한이 있다.
> 크레딧을 충전하면 무료 모델의 일일 요청 한도가 증가한다.

---

## 사용 가능한 모델 확인

### 웹 UI에서 확인

[https://openrouter.ai/models](https://openrouter.ai/models) 에서 전체 모델 목록 확인 가능
- 모델명, 제공사, 컨텍스트 길이, 입력/출력 토큰 단가 확인
- 필터로 무료 모델, 특정 카테고리 모델 선택 가능

### API로 모델 목록 조회

```python
import requests
import os

api_key = os.environ.get("OPENROUTER_API_KEY")

response = requests.get(
    url="https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

models = response.json()["data"]
for model in models[:10]:  # 상위 10개만 출력
    print(f"ID: {model['id']}, Context: {model.get('context_length', 'N/A')}")
```

### 주요 모델 ID 목록

| 제공사 | 모델 ID | 비고 |
|-------|---------|------|
| OpenAI | `openai/gpt-4o` | 최신 GPT-4o |
| OpenAI | `openai/gpt-4o-mini` | 저가형 GPT-4o |
| Anthropic | `anthropic/claude-3.5-sonnet` | Claude 3.5 Sonnet |
| Anthropic | `anthropic/claude-3-haiku` | 경량 Claude |
| Meta | `meta-llama/llama-3.1-8b-instruct` | Llama 3.1 8B (무료) |
| Meta | `meta-llama/llama-3.1-70b-instruct` | Llama 3.1 70B |
| Mistral | `mistral/mistral-7b-instruct` | Mistral 7B (무료) |
| Google | `google/gemini-flash-1.5` | Gemini Flash |
| Qwen | `qwen/qwen-2.5-72b-instruct` | Qwen 2.5 72B |
| DeepSeek | `deepseek/deepseek-r1` | DeepSeek R1 추론 모델 |

---

## 무료 모델 활용

OpenRouter는 무료로 사용 가능한 모델을 제공한다.
무료 모델 목록: [https://openrouter.ai/collections/free-models](https://openrouter.ai/collections/free-models)

### 무료 모델 식별 방법

- 모델 ID 끝에 `:free` 태그가 붙은 모델이 무료 모델
- 예: `meta-llama/llama-3.1-8b-instruct:free`

### 무료 모델 제한 사항

| 항목 | 제한 |
|------|------|
| 분당 요청 수 | 20 req/min |
| 일일 요청 수 | 200 req/day |
| 프로덕션 사용 | 권장하지 않음 |

### 자동 무료 모델 라우터 사용

`openrouter/free` 모델을 지정하면 자동으로 사용 가능한 무료 모델 중 요청에 맞는 모델로 라우팅된다.

```python
response = client.chat.completions.create(
    model="openrouter/free",
    messages=[{"role": "user", "content": "안녕하세요!"}]
)
```

---

## API 기본 사용법

### 엔드포인트 및 인증

| 항목 | 값 |
|------|-----|
| Base URL | `https://openrouter.ai/api/v1` |
| Chat Completions | `POST https://openrouter.ai/api/v1/chat/completions` |
| Models 조회 | `GET https://openrouter.ai/api/v1/models` |
| 인증 방식 | `Authorization: Bearer {API_KEY}` |
| Content-Type | `application/json` |

**선택적 헤더:**
- `HTTP-Referer`: 서비스 URL (대시보드 통계 집계용)
- `X-Title`: 서비스 이름 (대시보드 통계 집계용)

---

### Python - requests 라이브러리

```python
import requests
import json
import os

api_key = os.environ.get("OPENROUTER_API_KEY")

def call_openrouter(prompt: str, model: str = "meta-llama/llama-3.1-8b-instruct:free") -> str:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-service-url.com",  # 본인 서비스 URL로 변경
            "X-Title": "MyService"                            # 본인 서비스 이름으로 변경
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        })
    )
    result = response.json()
    return result["choices"][0]["message"]["content"]


# 사용 예
answer = call_openrouter("파이썬에서 리스트 컴프리헨션을 설명해주세요.")
print(answer)
```

---

### Python - openai 라이브러리

OpenRouter는 OpenAI API와 완전히 호환된다. `base_url`만 변경하면 기존 OpenAI 코드를 그대로 사용할 수 있다.

```bash
pip install openai python-dotenv
```

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://your-service-url.com",
        "X-Title": "MyService"
    }
)

# 기본 호출
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[
        {"role": "system", "content": "당신은 친절한 한국어 어시스턴트입니다."},
        {"role": "user", "content": "머신러닝과 딥러닝의 차이점을 설명해주세요."}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
print(f"\n사용 토큰 - 입력: {response.usage.prompt_tokens}, 출력: {response.usage.completion_tokens}")
```

---

### 스트리밍 응답 처리

대화 응답을 실시간으로 출력할 때 스트리밍을 사용한다.

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# 스트리밍 호출
stream = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[
        {"role": "user", "content": "재생에너지의 종류와 특징을 자세히 설명해주세요."}
    ],
    stream=True
)

# 청크 단위로 출력
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # 줄바꿈
```

**requests 라이브러리로 스트리밍:**

```python
import requests
import json
import os

api_key = os.environ.get("OPENROUTER_API_KEY")

def stream_response(prompt: str, model: str = "anthropic/claude-3.5-sonnet"):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }),
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                content = chunk["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)

    print()

stream_response("파이썬의 장점은 무엇인가요?")
```

---

## 지원 파라미터

OpenRouter는 OpenAI와 호환되는 파라미터 외에 추가 파라미터를 지원한다.

| 파라미터 | 범위 | 설명 |
|---------|------|------|
| `temperature` | 0 ~ 2 | 응답 창의성/무작위성 (기본값: 1) |
| `max_tokens` | 1 ~ context_length | 최대 출력 토큰 수 |
| `top_p` | 0 ~ 1 | 누적 확률 기반 샘플링 |
| `top_k` | 1 이상 | 상위 k개 토큰에서 샘플링 (OpenAI 모델 미지원) |
| `frequency_penalty` | -2 ~ 2 | 반복 단어 패널티 |
| `presence_penalty` | -2 ~ 2 | 새로운 주제 유도 패널티 |
| `repetition_penalty` | 0 ~ 2 | 반복 억제 (exclusive) |
| `seed` | 정수 | 재현 가능한 결과를 위한 시드 |
| `stream` | boolean | 스트리밍 응답 활성화 |

**OpenRouter 전용 파라미터 (`extra_body`):**

```python
response = client.chat.completions.create(
    model="meta-llama/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "provider": {
            "order": ["together", "fireworks"],  # 제공사 우선순위 지정
            "allow_fallbacks": True              # 폴백 허용
        }
    }
)
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

app = FastAPI()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://your-service.com",
        "X-Title": "MyAIService"
    }
)

DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


class ChatRequest(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    stream: bool = False


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
                {"role": "user", "content": request.message}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return {"response": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    def generate():
        stream = client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.message}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                data = {"content": chunk.choices[0].delta.content}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# 실행: uvicorn main:app --reload
```

---

### Spring Boot 연동

Spring Boot에서 OpenRouter를 사용하는 방법은 두 가지다.

#### 방법 1: Spring AI (권장)

Spring AI는 OpenAI 호환 엔드포인트를 지원하므로 `base-url`만 OpenRouter로 변경하면 된다.

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
spring:
  ai:
    openai:
      api-key: ${OPENROUTER_API_KEY}
      base-url: https://openrouter.ai/api/v1
      chat:
        options:
          model: anthropic/claude-3.5-sonnet
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
    public String chat(@RequestBody Map<String, String> request) {
        return chatClient.prompt()
            .user(request.get("message"))
            .call()
            .content();
    }
}
```

#### 방법 2: WebClient (직접 호출)

```java
@Service
public class OpenRouterService {

    private final WebClient webClient;

    public OpenRouterService(@Value("${openrouter.api-key}") String apiKey) {
        this.webClient = WebClient.builder()
            .baseUrl("https://openrouter.ai/api/v1")
            .defaultHeader("Authorization", "Bearer " + apiKey)
            .defaultHeader("Content-Type", "application/json")
            .defaultHeader("HTTP-Referer", "https://your-service.com")
            .build();
    }

    public String chat(String userMessage) {
        Map<String, Object> requestBody = Map.of(
            "model", "anthropic/claude-3.5-sonnet",
            "messages", List.of(
                Map.of("role", "user", "content", userMessage)
            )
        );

        Map response = webClient.post()
            .uri("/chat/completions")
            .bodyValue(requestBody)
            .retrieve()
            .bodyToMono(Map.class)
            .block();

        List<Map> choices = (List<Map>) response.get("choices");
        Map message = (Map) choices.get(0).get("message");
        return (String) message.get("content");
    }
}
```

---

## 멀티모달(이미지) 모델 호출

이미지를 지원하는 모델(GPT-4o, Claude 3 등)에 이미지를 함께 전달할 수 있다.

```python
import base64
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

# 이미지를 base64로 인코딩
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_data = encode_image("sample.png")

response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지에서 무엇을 볼 수 있나요?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

---

## 비용 모니터링 및 사용량 확인

### 대시보드에서 확인

- [https://openrouter.ai/activity](https://openrouter.ai/activity) 에서 API 호출 내역 및 비용 확인
- 모델별, 날짜별 사용량 통계 제공

### API 응답에서 토큰 사용량 확인

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": "Hello"}]
)

usage = response.usage
print(f"입력 토큰: {usage.prompt_tokens}")
print(f"출력 토큰: {usage.completion_tokens}")
print(f"총 토큰: {usage.total_tokens}")
```

### 생성 통계 조회 API

```python
import requests
import os

api_key = os.environ.get("OPENROUTER_API_KEY")
generation_id = "gen-xxxx"  # 응답의 id 필드 활용

response = requests.get(
    url=f"https://openrouter.ai/api/v1/generation?id={generation_id}",
    headers={"Authorization": f"Bearer {api_key}"}
)
print(response.json())
```

---

## 운영 환경 Best Practice

### 1. API 키 보안 관리

```python
# ❌ 코드에 직접 포함 금지
api_key = "sk-or-v1-xxxxxxxxxxxx"

# ✅ 환경 변수로 관리
import os
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY 환경 변수가 설정되지 않았습니다.")
```

### 2. 예외 처리 및 재시도 로직

```python
import time
import requests
import os

api_key = os.environ.get("OPENROUTER_API_KEY")

def call_with_retry(prompt: str, model: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            print(f"타임아웃 발생 (시도 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 지수 백오프

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate Limit
                time.sleep(60)
            elif response.status_code >= 500:  # 서버 오류
                time.sleep(5)
            else:
                raise e

    raise Exception(f"{max_retries}회 재시도 실패")
```

### 3. 모델 폴백 구성

```python
FALLBACK_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct"
]

def call_with_fallback(prompt: str) -> str:
    for model in FALLBACK_MODELS:
        try:
            return call_with_retry(prompt, model)
        except Exception as e:
            print(f"모델 {model} 실패: {e}")
    raise Exception("모든 모델 호출 실패")
```

### 4. 데이터 프라이버시 설정

기본적으로 OpenRouter는 프롬프트/응답을 로깅하지 않는다.
특정 제공사의 데이터 수집을 차단하려면:

```python
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    extra_body={
        "provider": {
            "data_collection": "deny"  # 제공사 데이터 수집 거부
        }
    }
)
```

---

## 자주 발생하는 오류 및 해결 방법

| 오류 | 원인 | 해결 방법 |
|------|------|-----------|
| `401 Unauthorized` | API 키 오류 또는 미설정 | API 키 재확인 및 환경 변수 설정 |
| `402 Payment Required` | 크레딧 부족 | 대시보드에서 크레딧 충전 |
| `429 Too Many Requests` | Rate Limit 초과 | 잠시 대기 후 재시도, 무료 모델 한도 확인 |
| `503 Service Unavailable` | 특정 제공사 장애 | 다른 모델로 폴백 또는 잠시 후 재시도 |
| `model not found` | 잘못된 모델 ID | `/api/v1/models` 로 사용 가능 모델 목록 확인 |
| `context_length_exceeded` | 입력이 컨텍스트 초과 | 입력 길이 줄이거나 더 큰 컨텍스트 모델 선택 |

---

> **참고 자료**
> - OpenRouter 공식 문서: https://openrouter.ai/docs
> - 모델 목록: https://openrouter.ai/models
> - 무료 모델 목록: https://openrouter.ai/collections/free-models
> - 사용량 확인: https://openrouter.ai/activity
