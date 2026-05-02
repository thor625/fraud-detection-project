from dotenv import load_dotenv
from pathlib import Path
import time
import boto3
import pandas as pd
import os
import io
from tqdm import tqdm
import pickle

# Load env once at module level
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

class Connectors:
    DATA_KEY = 'raw-data/creditcard.csv'
    LOCAL_CACHE = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw' / 'creditcard.csv'

    @staticmethod
    def connect_s3():
        key = os.getenv('AWS_ACCESS_KEY_ID')
        secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_DEFAULT_REGION')
        bucket = os.getenv('S3_BUCKET')

        if key and secret:
            print(f"Access key loaded: {key[:5]}...{key[-4:]}")
            print(f"Secret key loaded: {'*' * 20}")
            print(f"Region: {region}")
            print(f"Bucket: {bucket}")
        else:
            print("Credentials not found — check your .env file")
            return None

        print("\nVerifying AWS connection...")
        try:
            sts = boto3.client(
                'sts',
                aws_access_key_id=key,
                aws_secret_access_key=secret,
                region_name=region
            )
            identity = sts.get_caller_identity()
            print(f"Connected to AWS")
            print(f"Account ID: {identity['Account']}")
            print(f"User: {identity['Arn'].split('/')[-1]}")

            s3 = boto3.client(
                's3',
                aws_access_key_id=key,
                aws_secret_access_key=secret,
                region_name=region
            )
            return s3

        except Exception as e:
            print(f"✗ AWS connection failed: {e}")
            return None

    @staticmethod
    def load_data(s3_client):
        if Connectors.LOCAL_CACHE.exists():
            print("Local cache found — loading from disk...")
            start = time.time()
            df = pd.read_csv(Connectors.LOCAL_CACHE)
            elapsed = time.time() - start
            print(f"Loaded in {elapsed:.1f}s")
            return df

        print("No local cache — downloading from S3...")
        start = time.time()

        response = s3_client.head_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=Connectors.DATA_KEY
        )
        file_size = response['ContentLength']

        progress = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc='Downloading from S3',
            bar_format='{l_bar}{bar:30}{r_bar}'
        )

        buffer = []
        obj = s3_client.get_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=Connectors.DATA_KEY
        )

        for chunk in obj['Body'].iter_chunks(chunk_size=1024*1024):
            buffer.append(chunk)
            progress.update(len(chunk))

        progress.close()

        elapsed = time.time() - start
        raw_data = b''.join(buffer)
        df = pd.read_csv(io.BytesIO(raw_data))

        Connectors.LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(Connectors.LOCAL_CACHE, index=False)
        print(f"\nData loaded in {elapsed:.1f}s")
        print(f"Cached locally at {Connectors.LOCAL_CACHE}")
        return df
    
    @staticmethod
    def save_to_s3(obj, s3_client, key, bucket=os.getenv('S3_BUCKET')):
        print(f"Saving {key}...")
        
        # Serialize to buffer
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)
        file_size = buffer.getbuffer().nbytes

        # Upload with progress bar
        progress = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f'Uploading {key.split("/")[-1]}',
            bar_format='{l_bar}{bar:30}{r_bar}'
        )

        s3_client.upload_fileobj(
            buffer,
            bucket,
            key,
            Callback=lambda bytes_transferred: progress.update(bytes_transferred)
        )

        progress.close()
        print(f"Saved: {key}")

    @staticmethod
    def load_from_s3(s3_client, key, bucket=os.getenv('S3_BUCKET')):
        print(f"Loading {key}...")

        # Get file size first
        response = s3_client.head_object(Bucket=bucket, Key=key)
        file_size = response['ContentLength']

        progress = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f'Downloading {key.split("/")[-1]}',
            bar_format='{l_bar}{bar:30}{r_bar}'
        )

        buffer = io.BytesIO()

        s3_client.download_fileobj(
            bucket,
            key,
            buffer,
            Callback=lambda bytes_transferred: progress.update(bytes_transferred)
        )

        progress.close()
        buffer.seek(0)
        obj = pickle.loads(buffer.read())
        print(f"Loaded: {key}")
        return obj
    
    @staticmethod
    def load_model(s3_client, bucket=os.getenv('S3_BUCKET')):
        print("Loading model from S3...")
        obj = s3_client.get_object(
            Bucket=bucket,
            Key='model-artifacts/xgboost_model.pkl'
        )
        model = pickle.loads(obj['Body'].read())
        print("✓ Model loaded")
        return model

    @staticmethod
    def get_dynamodb():
        return boto3.resource(
            'dynamodb',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

    @staticmethod
    def save_prediction(dynamodb, result, table_name="fraud-predictions"):
        try:
            table = dynamodb.Table(table_name)
            table.put_item(Item=result)
        except Exception as e:
            print(f"DynamoDB write failed: {e}")
