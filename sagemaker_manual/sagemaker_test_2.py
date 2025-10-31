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