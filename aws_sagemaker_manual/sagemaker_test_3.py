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