import os
import boto3
from pathlib import Path
from botocore.exceptions import ClientError
from app.setting import R2_ACCOUNT_ID,R2_ACCESS_KEY_ID,R2_SECRET_ACCESS_KEY,R2_BUCKET_NAME,R2_ENDPOINT

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "indexes"


def download_indexes():
    required = {
        "R2_ACCOUNT_ID": R2_ACCOUNT_ID,
        "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
        "R2_BUCKET_NAME": R2_BUCKET_NAME,
    }
    #check if there is missing value 
    missing = [key for key,value in required.items() if not value]
    #if missing value then generate error 
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {','.join(missing)}"
        )

    #create R2/s3 client => used to connect with the r2 bucket
    s3 = boto3.client(
        "s3",
        endpoint_url = R2_ENDPOINT,
        aws_access_key_id = R2_ACCESS_KEY_ID,
        aws_secret_access_key = R2_SECRET_ACCESS_KEY,
        region_name = "auto"

    )

    #make sure the indexes folder exits 
    INDEX_DIR.mkdir(parents=True,exist_ok=True)

    #get all objets from r2 to the indeses folder
    paginator = s3.get_paginator("list_objects_v2")

    downloaded = 0

    for page in paginator.paginate(
        Bucket=R2_BUCKET_NAME,
        Prefix="indexes/",
    ):
        #for the file exact path we use object key => it finds the file exact path to download
        for obj in page.get("Contents", []):
            object_key = obj["Key"]

            # Ignore the indexes/ folder itself
            if object_key.endswith("/"):
                continue

            # Remove "indexes/" from the object key
            relative_path = object_key.removeprefix("indexes/")

            local_path = INDEX_DIR / relative_path

            # Create subdirectories if necessary
            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading: {object_key}")

            s3.download_file(
                R2_BUCKET_NAME,
                object_key,
                str(local_path),
            )

            downloaded += 1

    print(f"\nDownloaded {downloaded} index files.")
    print(f"Indexes location: {INDEX_DIR}")


if __name__ == "__main__":
    download_indexes()