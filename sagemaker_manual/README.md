## [주피터 노트북 소개]

컴퓨터와 소통하기 위해서는 개발 환경과 언어가 필요하다.    
다양한 언어와 툴이 있지만 그중에서도 가장 많이 사용되는 언어는 '파이썬', 가장 인기 있는 개발 환경은 '주피터 노트북'이다.    
주피터 노트북은 '파이썬'을 비롯해 다양한 개발 언어로 코드를 작성해 프로그래밍 할 수 있는 개발 환경이다.    
주피터 노트북은 파이썬을 코딩할 때 주로 사용하며 단계적으로 코드를 실행할 수 있어서 문서화/시각화/분석에 용이하다.   
<br/> 
프로그래밍 언어들 자체의 개발 환경이 있고, 또 다양한 개발 환경 플랫폼이 있지만,    
주피터 노트북을 많이 사용하는 이유는 웹 기반의 통합 개발 환경(IDE: Integrated Development Environment)이기 때문이다.   
통합 개발 환경(IDE)은 프로그래밍에 필요한 툴들이 하나의 인터페이스에 통합되어 있는 개발 환경을 말한다.   
주피터 노트북은 인터넷이 연결된 어느 컴퓨터에서나 웹 브라우저만 열면 접속 가능한 웹 기반 환경이다.    
그래서 어디서든 쉽게 사용 가능하다는 장점이 있고 게다가 오픈 소스라 무료이다.   
<br/> 
주피터 노트북은 '노트북'이라는 명칭에 걸맞게 대화형 모드를 지원하기에,    
코드를 한 줄 입력하면 실행되는 결과를 즉시 확인할 수 있다.    
에러가 발생해도 문제의 코드를 바로 수정하고, 실시간 피드백을 반영하면서 다음 코드를 이어 나갈 수 있다.   
주피터 노트북은 데이터 시각화에도 매우 용이하다.    
'마크다운' 기능으로 작성한 코드에 대해 설명을 추가할 수 있다.    
또, 데이터 분석이나 시각화에 활용되는 라이브러리를 사용해 불러온 데이터를 표, 그래프 등의 형태로 바로 시각화할 수 있다.   
이런 편의성 때문에 주피터 노트북은 데이터 사이언스는 물론, 머신러닝과 딥러닝 등 AI에도 많이 활용된다.   
<img width="800" alt="image1" src="https://github.com/user-attachments/assets/58c59714-00a2-475d-9b8e-a4e2d864f191">
<br/><br/>  

(1) 파이썬 설치 확인      
윈도우 cmd 창에서 "py" 명령어를 입력했을 때 아래 이미지와 같이 나오면 파이썬이 설치된 것이다.      
파이썬이 설치되어 있으면 주피터 노트북을 설치하기 위한 준비가 된 것이고    
만약 파이썬이 설치가 안 되어 있다면 구글 검색 등을 통해 파이썬을 먼저 설치하자.   
<img width="1600" alt="image2" src="https://github.com/user-attachments/assets/3855f7f9-e3c3-4ef4-b96d-0d2b5dcf6fcb">
<br/>

(2) 주피터 노트북 설치      
주피터 노트북을 설치하는 방법은 두 가지가 있다.    
하나는 cmd 창에서 설치하는 방법이고 하나는 아나콘다를 통해 설치하는 방법이다.   
아나콘다를 통해 설치하는 방법은 아나콘다도 깔아야 하고 시간이 더 걸리기 때문에   
여기서는 cmd 창에서 설치하는 방법을 설명하도록 하겠다.    
<br/>
윈도우 cmd 창에서 아래 명령어를 입력해 주피터 노트북을 설치하도록 한다.   
<br/>
명령어 : pip install jupyter
<img width="1600" alt="image3" src="https://github.com/user-attachments/assets/c4718cf4-d7b9-4e06-bbcd-4a76bda2acab">
<br/>

(3) 주피터 노트북 경로 설정   
설치가 완료되었다면 주피터 노트북 파일이 저장될 경로를 가정 먼저 설정해주어야 한다.   
주피터 노트북을 실행할 때 경로를 입력하지 않으면 현재 경로를 기준으로 실행된다.   
경로를 설정할 때는 아래의 명령어를 사용한다.   
<br/>
명령어 : jupyter notebook
<br/>

특정 경로를 저장 위치로 설정하고 싶으면 아래와 같이 설정한다.   
경로는 본인이 원하는대로 바꿔줄 수 있다.   
<br/>
명령어 : jupyter notebook --notebook-dir='C:\jupyter\notebook'
<img width="1600" alt="image4" src="https://github.com/user-attachments/assets/4da599f6-4da3-4511-9e92-61ff557b8647">
<br/>

경로 설정을 하고 나면 얼마 지나지 않아 웹브라우저에서 주피터 노트북이 실행된다.
<img width="1800" alt="image5" src="https://github.com/user-attachments/assets/f4078b59-2eee-4633-9338-b59da1c9f256">
<br/>

(4) 주피터 노트북 테스트   
이제 주피터 노트북을 테스트 해보도록 하겠다.   
"New -> Python 3 (jpykernel)"을 클릭해서 새로운 노트북을 생성한다.
<img width="1000" alt="image6" src="https://github.com/user-attachments/assets/a2d17ce6-5701-4fe4-9703-84d50842f4ca">
<br/>

ChatGPT 등의 생성형 AI를 통해 주피터 노트북에서 실행할 게임 코드를 알려달라고 하고 주피터 노트북에 코드를 복사하자.
<img width="800" alt="image7" src="https://github.com/user-attachments/assets/8a4998f9-8b7a-47b6-9b0e-5fbedb44f88e">
<img width="600" alt="image8" src="https://github.com/user-attachments/assets/3528127f-fbe5-4a60-870e-6b45d5f7893d">
<br/>

코드를 작성하고 실행할 때는 상단의 Run 버튼을 눌러도 되고, Shift+Enter(=Run)를 누르면 코드가 실행된다.
<img width="1000" alt="image9" src="https://github.com/user-attachments/assets/6f2a11cb-611f-4c87-805d-553f8c365459">
<img width="1000" alt="image10" src="https://github.com/user-attachments/assets/4e11ee8b-23c4-4616-a6bc-a8146a1fb6f0">
<br/><br/>

---
<br/>

## [Amazon 세이지메이커(SageMaker) 주피터 노트북 사용]

Amazon 세이지메이커 노트북 인스턴스는 주피터 노트북 인스턴스를 실행하는    
완전관리형 기계 학습(ML) EC2 인스턴스이다.   
이 노트북 인스턴스를 사용하여 데이터를 준비 및 처리하고   
기계 학습 모델을 훈련 및 배포하는데 사용할 수 있는 주피터 노트북을 생성하고 관리한다.   
<br/>

1. AWS 콘솔에서 Amazon 세이지메이커 서비스에 들어간다.      
그 후 "Notebooks" 항목을 클릭한다.   
<img width="1800" alt="image" src="https://github.com/user-attachments/assets/d10abca7-ab1f-4033-8c19-aa595349c1da">
<br/><br/>   

2. "노트북 인스턴스 생성"에서 지정할 수 있는 것들은 아래와 같다.      
기본적으로 노트북 인스턴스 이름과 유형, 볼륨 크기, 권한, 네트워크는 지정해주자.    
<br/>

(1) 노트북 인스턴스 이름   
- 노트북 인스턴스 이름에 특정 인스턴스 이름을 입력   
- 최대 63자의 영숫자   
- 하이픈(-)을 포함 가능   
- 공백은 포함 안됨   
- AWS 리전의 계정 내에서 고유해야함
<br/> 
  
(2) 노트북 인스턴스 유형   
- t 패밀리 (표준)   
- c패밀리 (컴퓨팅 최적화)   
- p패밀리 (엑셀러레이티드 컴퓨팅)   
<br/>
  
(3) 노트북 권한   
- 노트북 인스턴스에는 SageMaker 및 S3 같은 다른 서비스를 호출할 수 있는 권한이 필요   
- SageMaker 리소스에 액세스하는데 필요한 권한을 보유한 계정의 기존 IAM 역할 및 새 역할 생성 선택
<br/>   
  
(4) 노트북 네트워크 환경   
- VPC   
- 서브넷   
- 보안그룹   
<img width="800" alt="image2" src="https://github.com/user-attachments/assets/c926edb0-fa04-4415-ba45-1e4a42840ed2">
<br/><br/>

3. 이제 노트북 인스턴스를 프로비저닝하는 과정을 잠시 기다리도록 하자.    
<img width="1800" alt="image3" src="https://github.com/user-attachments/assets/8e695c14-03a0-422c-847d-7c84992853e1">
<br/><br/>

4. 노트북 인스턴스 상태가 "InService"로 바뀌면 주피터나 주피터랩을 실행하면 된다.   
<img width="1800" alt="image4" src="https://github.com/user-attachments/assets/4da51d1b-4f1c-4b6b-af66-9e815e77d546">
<br/><br/>

[Jupyter와 Jupyter Lab의 차이]   
(1) Jupyter   
- cell 단위로 실행할 수 있어서 짧은 코드를 테스트 할 때 편리   
- 에러 위치를 쉽게 확인할 수 있음   
- 무거운 IDE가 필요 없어 jupyter 커널을 열고 웹브라우저를 열면 바로 작성할 수 있음   
- 데이터 과학, 머신러닝, 딥러닝 프로젝트를 할 때 많이 사용되는 개발환경
<br/><br/>
  
(2) Jupyter Lab   
- 주피터 노트북을 기반으로 만들어진 차세대 웹기반 사용자 인터페이스   
- 기존의 노트북은 탭 하나당 하나의 노트북 파일만 열 수 있으나 랩은 다중 창을 지원   
- Jupyter notebook과 terminal도 편리하게 사용이 가능   
<br/>  

5. Jupyter를 열고 New를 클릭하면 여러 가지 항목들이 보인다.   
일단 테스트를 위해서는 conda_python3를 클릭해보자.    
그 외의 항목들에 대한 설명은 아래와 같다.
<br/>

(1) R : R 프로그래밍 언어를 사용하여 노트북을 생성할 수 있는 커널입니다.   
주로 통계 분석과 데이터 시각화에 사용된다.      
<br/>
(2) Sparkmagic (PySpark) : PySpark를 사용하여 아파치 스파크와 상호 작용할 수 있는 커널이다.       
PySpark는 스파크의 파이썬 API로, 대규모 데이터 처리를 위한 분산 컴퓨팅 시스템이다.      
<br/> 
(3) Sparkmagic (Spark) : 아파치 스파크의 스칼라(Scala) API와 상호 작용할 수 있는 커널이다.       
스파크는 대규모 데이터 분석과 처리를 위한 분산 컴퓨팅 시스템이다.      
<br/>
(4) Sparkmagic (SparkR) : 스파크의 R API와 상호 작용할 수 있는 커널이다.       
이는 R을 사용하여 분산 데이터 처리를 수행할 수 있게 해준다.      
<br/>
(5) conda_python3 : Python 3을 사용하여 노트북을 생성할 수 있는 커널이다.       
이는 기본적인 파이썬 환경을 제공하며, 데이터 과학, 머신 러닝, 일반적인 파이썬 프로그래밍에 사용된다.      
<br/>
(6) conda_pytorch_p310 : Python 3.10 환경에서 PyTorch를 사용하여 노트북을 생성할 수 있는 커널이다.       
PyTorch는 머신 러닝과 딥 러닝 프레임워크로, 특히 인공 신경망 연구와 개발에 널리 사용된다.      
<br/>
(7) conda_tensorflow2_p310 : Python 3.10 환경에서 TensorFlow 2를 사용하여 노트북을 생성할 수 있는 커널이다.       
TensorFlow는 딥 러닝 및 머신 러닝 모델을 구축하고 배포하는 데 사용되는 오픈 소스 라이브러리다.      
<br/>
(8) Text File : 일반 텍스트 파일을 생성한다.   
<br/>
(9) Folder : 새 폴더를 생성한다.   
<br/>
(10) Terminal : 주피터 노트북 환경 내에서 터미널을 연다.          
이를 통해 명령줄 작업을 수행할 수 있다.         
![image5](https://github.com/user-attachments/assets/a16fd2f9-e383-4a14-a1db-8b65d601226c)
<br/><br/>

7. ChatGPT에서 간단한 코드를 받아 복사한 후 SageMaker Jupyter에서 코드를 실행해보자. 
<img width="800" alt="image6" src="https://github.com/user-attachments/assets/77e8ee66-0256-43c8-9f47-b51a05848063">
<img width="600" alt="image7" src="https://github.com/user-attachments/assets/7de82b0d-b51a-4db0-a19f-7a5d521a0c5a">
<br/><br/>

8. 로컬 PC에서 주피터 노트북을 실행했을 때와 동일하게 SageMaker 노트북에서도 코드가 정상 동작하는 것을 확인할 수 있다.
<img width="1000" alt="image8" src="https://github.com/user-attachments/assets/923a0a65-126b-48c6-822b-f6660a2a285c">
<br/><br/>

---
<br/>

## [로컬 PC 주피터 노트북과 SageMaker 노트북의 차이점]

(1) 로컬 PC는 사용자가 직접 파이썬 환경, 라이브러리, 종속성 등을 설치하고 관리해야 한다.    
환경 설정 및 관리를 위해 추가적인 시간이 소요될 수 있다.   
하지만 SageMaker 노트북은 AWS에서 제공하는 관리형 서비스로 노트북 인스턴스를 클릭 몇 번으로 쉽게 생성할 수 있다.    
필요한 라이브러리와 프레임워크가 미리 설치된 다양한 커널을 제공한다.   
<br/>

(2) 로컬 PC : 로컬 컴퓨터의 하드웨어 자원(메모리, CPU, GPU)에 의존한다.    
복잡한 머신러닝 모델을 훈련할 때 자원이 부족할 수 있다.   
SageMaker: 다양한 인스턴스 타입을 선택하여 필요에 따라 CPU, GPU, 메모리를 확장할 수 있다.    
대규모 데이터 처리와 복잡한 모델 훈련에 적합하다.   
<br/>

(3) 로컬 PC : 하드웨어 확장이 어렵고 비용이 많이 든다.   
SageMaker : 클라우드 기반 서비스로 필요에 따라 쉽게 확장할 수 있다.    
여러 인스턴스를 병렬로 사용할 수 있어 확장성이 뛰어나다.   
<br/>

(4) 로컬 PC : 데이터가 로컬에 저장되며, 네트워크를 통한 데이터 전송이 필요하다.    
보안 설정과 데이터 백업을 직접 관리해야 한다.   
SageMaker : S3와 통합되어 대규모 데이터를 쉽게 접근하고 저장할 수 있다.    
AWS의 보안 기능을 통해 데이터 보안을 강화할 수 있다.   
<br/><br/>

## [Amazon SageMaker 노트북의 장점]

(1) 편리한 설정 및 관리 : SageMaker는 클릭 몇 번으로 머신러닝 환경을 설정할 수 있다.    
다양한 미리 설정된 환경을 제공하여 빠르게 시작할 수 있다.   
노트북 인스턴스를 생성하고 관리하는 데 드는 시간과 노력을 줄일 수 있다.   
<br/>

(2) 강력한 컴퓨팅 자원 :   
다양한 인스턴스 타입(CPU, GPU, 메모리)을 선택할 수 있어, 복잡한 머신러닝 모델의 훈련 및 예측을 효율적으로 수행할 수 있다.   
필요에 따라 인스턴스를 시작하거나 중지하여 비용 효율적으로 자원을 관리할 수 있다.   
<br/>

(3) 자동화 및 배포 기능 :   
SageMaker는 모델 훈련, 튜닝, 배포를 자동화할 수 있는 다양한 도구를 제공한다.    
훈련된 모델을 쉽게 배포하고, 엔드포인트를 통해 실시간 예측 서비스를 제공할 수 있다.   
<br/>

(4) 데이터 접근 및 통합 :   
AWS의 다양한 데이터 저장소(S3, RDS, DynamoDB 등)와 쉽게 통합할 수 있다.    
대규모 데이터를 처리하고 분석하는 데 용이하다.   
데이터 파이프라인을 구축하고 관리하는데 필요한 도구들을 제공한다.   
<br/>

(5) 협업 및 공유 :   
SageMaker 노트북은 팀원 간의 협업을 쉽게 할 수 있도록 노트북을 공유하는 기능을 제공한다.    
여러 사용자가 동시에 같은 노트북에서 작업할 수 있다.   
<br/>

(6) 보안 및 규정 준수 :   
AWS의 보안 인프라를 통해 데이터와 애플리케이션을 보호할 수 있다.    
VPC, IAM, KMS 등의 기능을 활용하여 보안을 강화할 수 있다.   
다양한 규정 준수 인증을 통해, 산업 표준을 준수하는 머신러닝 워크플로우를 구축할 수 있다.   
<br/><br/>

---
<br/>

## [세이지메이커(SageMaker) & 베드락(Bedrock) 연동 머신러닝]

이제 베드락과 세이지메이커를 연동해보는 테스트를 해보겠다.
<br/>

1. 세이지메이커 주피터 노트북에서 아래 코드(sagemaker_test_1.py)를 실행해보자.
<br/><br/>   

```python
import boto3
import json
brt = boto3.client(service_name='bedrock-runtime')


body = json.dumps({
    "prompt": "\n\nHuman: explain black holes to university student\n\nAssistant:",
    "max_tokens_to_sample": 500,
    "temperature": 0.1,
    "top_p": 0.9,
})


modelId = 'anthropic.claude-v2'
accept = 'application/json'
contentType = 'application/json'


response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)


response_body = json.loads(response.get('body').read())


# text
print(response_body.get('completion'))
```
<br/>
  
2. 기본 권한 상태에서 위의 코드를 세이지메이커 주피터 노트북에서 실행하면 아래 이미지와 같은 에러가 발생할 것이다.    
기본 권한 상태에서는 베드락 AI 모델에 대한 접근 권한이 없기 때문에 발생하는 에러다.   
<img width="1000" alt="image1" src="https://github.com/user-attachments/assets/f4b83e9c-f306-4564-86e2-2f74dd252cd0">
<br/><br/>

3. 세이지메이커 주피터 노트북 생성시 지정했던 IAM 역할에 권한을 추가해보자.
<img width="1400" alt="image2" src="https://github.com/user-attachments/assets/0561f1d5-921b-4d94-8850-054e8edb0acc">
<br/><br/>
  
4. 베드락 AI 모델을 호출하는 권한만 넣어주어도 되지만 본 테스트에서는 일단 "AmazonBedrockFullAccess" 권한을 추가하도록 하겠다. 
<img width="1800" alt="image3" src="https://github.com/user-attachments/assets/2c037a9c-3040-4df3-84ac-ce82c9a6cb82">
<br/><br/>
         
5. 다시 주피터 노트북에서 코드를 실행하면 정상적으로 코드가 실행되어 결과가 출력되는 것을 확인할 수 있다.
<img width="1000" alt="image4" src="https://github.com/user-attachments/assets/b0657e81-1282-423f-bac8-c834de7d7170">
<br/><br/>
  
6. 이제 S3에 세이지메이커 주피터 노트북 코드 실행 결과를 저장해보겠다.   
우선 주피터 노트북의 코드 실행 결과를 저장할 S3를 확인해야 한다.   
지금까지의 과정을 따라왔다면 아마 아래와 같이 세이지메이커 S3가 만들어져 있을 것이다.   
<img width="800" alt="image5" src="https://github.com/user-attachments/assets/c4425162-ad4d-4b86-9492-5f849bb8abd9">
<br/><br/>

7. 다음은 주피터 노트북의 코드를 아래 코드(sagemaker_test_2.py)와 같이 수정해본다.   
sagemaker_test_2.py 코드 내용 중 'user-sagemaker-bucket' 부분은 실제 세이지메이커 버킷명으로 수정해줘야 한다.   
<br/>

```python
import boto3
import json
import datetime
import pandas as pd


# AWS S3 및 Bedrock 클라이언트 생성
s3 = boto3.client('s3')
brt = boto3.client(service_name='bedrock-runtime')


# Bedrock 모델 호출 함수
def invoke_bedrock_model(prompt):
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 500,
        "temperature": 0.1,
        "top_p": 0.9,
    })


    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'


    response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())


    return response_body


# S3에 데이터 저장 함수
def save_to_s3(data, bucket_name, file_name):
    s3.put_object(Body=json.dumps(data), Bucket=bucket_name, Key=file_name)


# S3에서 데이터 불러오기 함수
def load_data_from_s3(bucket_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    data_frames = []
    for obj in response.get('Contents', []):
        response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
        data = json.loads(response['Body'].read().decode('utf-8'))
        df = pd.json_normalize(data)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


# 메인 실행 함수
def main():
    # 사용자 입력 프롬프트
    prompt = "\n\nHuman: explain black holes to university student\n\nAssistant:"
    

    # Bedrock 모델 호출
    response_body = invoke_bedrock_model(prompt)
    

    # 결과 출력
    completion_text = response_body.get('completion')
    print(completion_text)
    

    # S3에 데이터 저장
    bucket_name = 'user-sagemaker-bucket'  # 사용자의 S3 버킷 이름으로 변경
    file_name = f'results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    save_to_s3(response_body, bucket_name, file_name)
    

    # S3에서 데이터 불러오기
    prefix = 'results_'  # 파일 이름의 접두사
    data_df = load_data_from_s3(bucket_name, prefix)
    

    # 불러온 데이터 출력 (필요한 부분만 출력)
    print('Data loaded from S3:')
    for index, row in data_df.iterrows():
        print(row['completion'])


# 메인 함수 실행
if __name__ == "__main__":
    main()    
```
<br/>

8. 세이지메이커 주피터 노트북에서 출력 결과가 정상적으로 나오는 것을 확인한 후 S3에도 정상적으로 파일이 추가되었는지 확인한다.
<img width="1000" alt="image6" src="https://github.com/user-attachments/assets/8fffb929-dc9a-4de9-b33a-a299bfdc9e35">
<img width="1600" alt="image7" src="https://github.com/user-attachments/assets/052da0ec-67cf-415f-8fbf-f47faefaddfc">
<br/><br/>
  
9. S3에 파일이 정상적으로 추가된 것이 확인되면 다운받아 파일 내용을 확인해보자.   
앞으로 세이지메이커의 주피터 노트북에서 코드가 실행될 때마다 S3에 이런 파일이 추가될 것이다.
<img width="1400" alt="image8" src="https://github.com/user-attachments/assets/42a02590-d9ae-4baf-bbd9-183f0c3b8be5">
<img width="1600" alt="image9" src="https://github.com/user-attachments/assets/327f2e68-cd70-4ae5-a0dc-aed6cfb08ef8">
<br/><br/>

10. 세이지메이커 주피터 노트북에서 같은 코드를 계속 실행해도 답변이 바뀌는 것을 알 수 있다.   
S3에 축적된 파일의 내용을 종합하여 답변을 점점 발전시키는 것이다.
<img width="2000" alt="image10" src="https://github.com/user-attachments/assets/e1d1d4fc-ef60-44be-895d-a5abda03e5a0">
<img width="2000" alt="image11" src="https://github.com/user-attachments/assets/a20027ad-2d36-4b23-b577-21189a496bb4">
<br/><br/>

11. 사람이 수동으로 S3에 "results_"라는 이름의 파일을 집어넣고   
그 파일에 나와있는 내용까지 합쳐서 머신러닝을 한 다음 주피터 노트북에서 결과를 출력할 수 있다.      
또한, 세이지 메이커 주피터 노트북의 출력 결과 역시 S3에 저장하고 해당 S3를 바탕으로 머신러닝을 하여 답변을 더욱 발전시킬 수 있다.   
세이지메이커 주피터 노트북에서 코드를 아래 코드(sagemaker_test_3.py)와 같이 작성해보자.   
sagemaker_test_3.py 코드 내용 중 'user-sagemaker-bucket' 부분은 실제 세이지메이커 버킷명으로 수정해줘야 한다.
<br/>

```python
import boto3
import json
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# AWS S3 클라이언트 생성
s3 = boto3.client('s3')


# S3에서 데이터 불러오기 함수
def load_data_from_s3(bucket_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    data_frames = []
    for obj in response.get('Contents', []):
        response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
        data = json.loads(response['Body'].read().decode('utf-8'))
        df = pd.json_normalize(data)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


# Bedrock 모델 호출 함수
def invoke_bedrock_model(prompt):
    brt = boto3.client(service_name='bedrock-runtime')
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 500,
        "temperature": 0.1,
        "top_p": 0.9,
    })


    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'


    response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())


    return response_body


# S3에 데이터 저장 함수
def save_to_s3(data, bucket_name, file_name):
    s3.put_object(Body=json.dumps(data), Bucket=bucket_name, Key=file_name)


# 데이터 전처리 함수 (예제)
def preprocess_data(df):
    # 예제: 단순히 텍스트 길이를 feature로 사용
    df['text_length'] = df['completion'].apply(len)
    return df[['text_length']], df['text_length']  # 예측 대상은 자기 자신 (단순 예제)


# 머신러닝 모델 훈련 및 예측 함수
def train_and_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


# 메인 실행 함수
def main():
    bucket_name = 'user-sagemaker-bucket'  # 사용자의 S3 버킷 이름으로 변경
    prefix = 'results_'  # 파일 이름의 접두사


    # S3에서 데이터 불러오기
    data_df = load_data_from_s3(bucket_name, prefix)
    

    # 사용자 입력 프롬프트
    prompt = "\n\nHuman: explain black holes to university student\n\nAssistant:"
    

    # Bedrock 모델 호출
    response_body = invoke_bedrock_model(prompt)
    

    # 결과 출력
    completion_text = response_body.get('completion')
    print(completion_text)
    

    # S3에 데이터 저장
    file_name = f'results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    save_to_s3(response_body, bucket_name, file_name)
    

    # S3에서 불러온 데이터에 새로 생성된 데이터를 추가
    new_row = pd.DataFrame([response_body])
    data_df = pd.concat([data_df, new_row], ignore_index=True)
    

    # 데이터 전처리
    X, y = preprocess_data(data_df)
    

    # 머신러닝 모델 훈련 및 예측
    predictions = train_and_predict(X, y)
    

    # 예측 값 출력 제거 (순수한 답변만 출력)
    #for prediction in predictions:
    #    print(f'Predicted text length: {prediction}')


# 메인 함수 실행
if __name__ == "__main__":
    main()    
```
<br/>
    
12. 세이지메이커 주피터 노트북에서 정상적으로 답변이 출력되는 것을 확인하자.   
그 뒤에 코드를 계속 반복 실행하면서 S3에도 정상적으로 파일이 추가되는지 확인한다.
<img width="1000" alt="image12" src="https://github.com/user-attachments/assets/e05ab652-841b-437d-bd35-16c455242731">
<img width="1600" alt="image13" src="https://github.com/user-attachments/assets/2337bcc1-c75e-452a-a399-2efd97cdaf64">
