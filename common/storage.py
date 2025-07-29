import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

def upload_directory_to_minio(local_dir: str, remote_prefix: str):
    """
    Recursively uploads all files in a local directory to MinIO with the given prefix.
    """
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            remote_path = os.path.join(remote_prefix, relative_path).replace("\\", "/")

            with open(local_path, "rb") as f:
                file_content = f.read()
                default_storage.save(remote_path, ContentFile(file_content))
