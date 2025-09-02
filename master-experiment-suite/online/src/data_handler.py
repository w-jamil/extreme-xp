import requests
import zipfile
import os
import shutil
import time
import json

def prepare_data_from_zenodo(zenodo_archive_url, target_dir):
    """
    Handles the entire data acquisition process from Zenodo with a highly robust,
    on-disk streaming download to prevent memory and network timeout issues.
    """
    if os.path.exists(target_dir) and any(f.endswith('.parquet') for f in os.listdir(target_dir)):
        print(f"--> Data found locally in '{target_dir}'. Skipping download.")
        return True

    print(f"--> Local data not found. Preparing to download from Zenodo...")
    
    # Extract record ID from URL
    if '/api/records/' in zenodo_archive_url:
        # Format: https://zenodo.org/api/records/13787591/files-archive
        record_id = zenodo_archive_url.split('/api/records/')[1].split('/')[0]
    elif '/records/' in zenodo_archive_url:
        # Format: https://zenodo.org/records/13787591
        record_id = zenodo_archive_url.split('/records/')[1].split('?')[0].split('/')[0]
    else:
        print(f"--> ERROR: Unable to extract record ID from URL: {zenodo_archive_url}")
        return False
    
    print(f"    Record ID: {record_id}")

    try:
        # Get record metadata to find individual files
        api_url = f"https://zenodo.org/api/records/{record_id}"
        print(f"--> Fetching record metadata from: {api_url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        record_data = response.json()
        
        # Extract parquet file URLs
        parquet_files = []
        for file_info in record_data.get('files', []):
            if file_info['key'].endswith('.parquet'):
                parquet_files.append({
                    'name': file_info['key'],
                    'url': file_info['links']['self'],
                    'size': file_info['size']
                })
        
        print(f"--> Found {len(parquet_files)} parquet files to download")
        
        if not parquet_files:
            print("--> ERROR: No parquet files found in the Zenodo record")
            return False
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Download each parquet file individually
        for i, file_info in enumerate(parquet_files):
            print(f"--> Downloading {i+1}/{len(parquet_files)}: {file_info['name']}")
            print(f"    Size: {file_info['size'] / 1024**2:.2f} MB")
            print(f"    URL: {file_info['url']}")
            
            output_path = os.path.join(target_dir, file_info['name'])
            
            # Download file
            with requests.get(file_info['url'], stream=True, headers=headers, timeout=60) as r:
                r.raise_for_status()
                total_size = file_info['size']
                downloaded_size = 0
                
                with open(output_path, 'wb') as f:
                    start_time = time.time()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Print progress every 2 seconds
                            if time.time() - start_time > 2:
                                progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                                print(f"      Progress: {downloaded_size / 1024**2:.2f} / {total_size / 1024**2:.2f} MB ({progress:.1f}%)", end='\r')
                                start_time = time.time()
                
                print(f"\n      ✓ Downloaded: {file_info['name']}")
        
        print(f"\n--> Successfully downloaded all {len(parquet_files)} parquet files to '{target_dir}'")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: A network error occurred. Please check the URL and your internet connection. Details: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"\nERROR: Failed to parse Zenodo API response. Details: {e}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during data preparation: {e}")
        return False
        
