import io
from minio import Minio
from minio.error import S3Error

class MinioStorageService:
    def __init__(
            self,
            endpoint: str = "127.0.0.1:9000",
            access_key: str = "minioadmin",
            secret_key: str = "minioadmin",
            bucket_name: str = "rag-documents",
            secure: bool = False
    ):
        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as e:
            print(f"Warning MinIO bucket check: {e}")

    def upload_file(self, filename: str, file_bytes: bytes, content_type: str = "application/octet-stream") -> str:
        file_stream = io.BytesIO(file_bytes)
        file_length = len(file_bytes)

        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=filename,
            data=file_stream,
            length=file_length,
            content_type=content_type
        )
        return filename
