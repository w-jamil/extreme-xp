import requests
import zipfile
import os
import shutil
import time

def prepare_data_from_zenodo(zenodo_archive_url, target_dir):
    """
    Handles the entire data acquisition process from Zenodo with a highly robust,
    on-disk streaming download to prevent memory and network timeout issues.
    """
    if os.path.exists(target_dir) and any(f.endswith('.parquet') for f in os.listdir(target_dir)):
        print(f"--> Data found locally in '{target_dir}'. Skipping download.")
        return True

    print(f"--> Local data not found. Preparing to download from Zenodo...")
    print(f"    URL: {zenodo_archive_url}")

    # Use a temporary file to save the download stream
    temp_zip_path = 'temp_download.zip'

    try:
        # --- START OF THE ROBUST DOWNLOAD FIX ---
        
        # Download the file in chunks and save directly to disk
        with requests.get(zenodo_archive_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0
            
            print(f"--> Connection established. Total file size: {total_size / 1024**2:.2f} MB")
            print("--> Downloading data to temporary file...")

            with open(temp_zip_path, 'wb') as f:
                start_time = time.time()
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Optional: Print progress
                    if time.time() - start_time > 2: # Print every 2 seconds
                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        print(f"    Downloaded {downloaded_size / 1024**2:.2f} / {total_size / 1024**2:.2f} MB ({progress:.1f}%)", end='\r')
                        start_time = time.time()
        
        print("\n--> Download complete. Unzipping relevant files from archive...")
        # --- END OF THE ROBUST DOWNLOAD FIX ---
        
        os.makedirs(target_dir, exist_ok=True)
        
        # Now, open the zip file from the disk
        with zipfile.ZipFile(temp_zip_path) as z:
            for member in z.infolist():
                if member.filename.endswith('.parquet') and not member.is_dir():
                    base_filename = os.path.basename(member.filename)
                    output_path = os.path.join(target_dir, base_filename)
                    
                    with z.open(member) as source, open(output_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    print(f"    - Extracted: {base_filename}")
        
        print(f"--> Successfully extracted all .parquet files to '{target_dir}'")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: A network error occurred. Please check the URL and your internet connection. Details: {e}")
        return False
    except zipfile.BadZipFile:
        print("\nERROR: The downloaded file is not a valid zip archive. It may be incomplete.")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during data preparation: {e}")
        return False
    finally:
        # --- CLEANUP: Always remove the temporary zip file ---
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
            print("--> Temporary download file cleaned up.")