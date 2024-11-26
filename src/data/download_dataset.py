import requests
import zipfile
import os

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Downloaded file to {output_path}")

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Unzipped file to {extract_to}")

def delete_zipfile(zip_file_path):
    os.remove(zip_file_path)
    print(f"Deleted downloaded zip file: {zip_file_path}")

def download_dataset():
    # URL of the zip file
    url = "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
    zip_file_path = os.path.abspath("data/raw/dataset.zip")
    extracted_folder = os.path.abspath("data/raw/")
    dataset_original_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Ensure the extracted folder exists
    os.makedirs(extracted_folder, exist_ok=True)

    # Download and unzip the file
    download_file(url, zip_file_path)
    unzip_file(zip_file_path, extracted_folder)
    delete_zipfile(zip_file_path)

if __name__ == "__main__":
    download_dataset()