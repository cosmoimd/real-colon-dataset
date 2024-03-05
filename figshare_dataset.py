#!/usr/bin/env python

""" Download the REAL-Colon dataset from figshare and extract all the files

    Usage:
        - Update download_dir = './dataset/' with path to the folder to download the REAL-colon dataset
        - python3 figshare_dataset.py

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""
import os
import requests
import tarfile
import time
from multiprocessing import Pool

# Define Figshare article URL and API endpoint
article_url = 'https://api.figshare.com/v2/articles/22202866'

# Specify the path to your custom CA bundle here, or set to None to use the default CA bundle
custom_ca_bundle = None

# Specify the path where to download the dataset
DOWNLOAD_DIR = "./dataset/"


# Helper function to return the file size if a path exists, -1 otherwise
def file_exists(local_filename):
    if os.path.exists(local_filename):
        return os.path.getsize(local_filename)
    return -1


def download_file(args):
    url, local_filename = args
    simple_filename = local_filename
    existing_file_size = file_exists(local_filename)

    # Check if file already exists
    if existing_file_size != -1:
        print(f'{simple_filename} already exists.')

    max_attempts = 1000  # Maximum number of download attempts
    attempt = 0
    retry_delay = 180  # Wait for 3 minutes (180 seconds) before retrying

    while attempt < max_attempts:
        try:
            # get request using the CA verification if provided
            with requests.get(url, stream=True, verify=custom_ca_bundle or True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded_size = 0
                start_time = time.time()

                # Download the file from figshare and record the size and time it took
                with open(local_filename, 'wb') as f:
                    print(f'Downloading {simple_filename}... {total_size / (1024 * 1024):.2f} MB')
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                    elapsed_time = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed_time), 60)
                    print(
                        f'Downloaded {simple_filename}...'
                        f' {downloaded_size / (1024 * 1024):.2f} '
                        f'MB/{total_size / (1024 * 1024):.2f} MB downloaded in '
                        f'{minutes}m {seconds}s', end='\r')
                print()  # Print a newline after download completion
            return local_filename

        # If there is an error, wait for 180 seconds and try to continue where you left off
        except Exception as e:
            error_message = str(e)
            if 'IncompleteRead' in error_message:
                print(f'Connection error occurred: {error_message}. Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)  # Wait for the specified delay
                attempt += 1
            else:
                print(f'An unexpected error occurred: {error_message}. Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)  # Wait for the specified delay
                attempt += 1

    print(f'Failed to download {simple_filename} after {max_attempts} attempts.')
    return None


def extract_file(args):
    file_path, download_dir = args

    # If there is a compressed .tar.gz file and the unzipped directory doesn't already exist, then extract it
    if file_path.endswith('.tar.gz'):
        file_comp_path = os.path.join(download_dir, file_path)
        file_name = file_path.rstrip('.tar.gz')
        extracted_folder_name = os.path.splitext(os.path.basename(file_name))[0]
        extracted_folder_path = os.path.join(download_dir, extracted_folder_name)

        if not os.path.exists(extracted_folder_path):
            print(f'Extracting {file_path}...')
            with tarfile.open(file_comp_path, 'r') as tar_ref:
                tar_ref.extractall(download_dir)

        # Delete the tar.gz file ~
        print(f'Deleting {file_path}...')
        os.remove(file_comp_path)


def extract_files(file_paths, download_dir):
    # Control the number of processes for extraction
    num_processes = 3  # Change this to the desired number of extraction processes
    num_processes = min(num_processes, len(file_paths))

    # Create a pool of worker processes for extraction
    pool = Pool(processes=num_processes)

    # Map the extraction function to the file paths
    pool.map(extract_file, [(file_path, download_dir) for file_path in file_paths])


def main():
    response = requests.get(article_url, verify=custom_ca_bundle or True)
    response.raise_for_status()
    article_data = response.json()

    # Change the download_dir to the directory you want the downloads in
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    download_tasks = []  # Store download tasks as tuples (url, local_file_path)

    for file_info in article_data['files']:

        # Get the file names
        file_url = file_info['download_url']
        file_name = file_info['name']
        local_file_path = os.path.join(DOWNLOAD_DIR, file_name)
        local_dir_check = local_file_path.rstrip('.tar.gz')

        # Check if either the directory or the zipped file already exists
        existing_file_size = file_exists(local_file_path)
        existing_file_path = file_exists(local_dir_check)
        if existing_file_path != -1:
            print(f'{local_dir_check} already exists. Skipping download.')
            continue
        if existing_file_size != -1:
            remote_file_size = int(file_info['size'])
            if existing_file_size == remote_file_size:
                print(f'{file_name} already exists. Skipping download.')
                continue
            else:
                print(f'{file_name} already exists but has a different size. Deleting it...')
                os.remove(local_file_path)

        download_tasks.append((file_url, local_file_path))
        print(f'Queued {file_name} for download...')

    # Control the number of processes by adjusting this variable
    num_processes = 4  # Change this to the desired number of processes

    # Ensure the number of processes does not exceed the number of tasks
    num_processes = min(num_processes, len(download_tasks))

    # Create a pool of worker processes and download files concurrently
    if num_processes != 0:
        pool = Pool(processes=num_processes)
        downloaded_files = pool.map(download_file, download_tasks)

    # Now that all downloads are complete, extract the files
    tar_files = [file for file in os.listdir(DOWNLOAD_DIR) if file.endswith('.tar.gz')]
    extract_files(tar_files, DOWNLOAD_DIR)

    print('Process completed.')


if __name__ == "__main__":
    main()
