import requests
import zipfile
import os
import io

def prepare_data_from_zenodo(zenodo_archive_url, target_dir):
    """
    Checks for data locally; if not found, downloads a zip archive from Zenodo,
    finds all .parquet files within it (regardless of sub-directory), and
    extracts them directly into the target directory.

    Args:
        zenodo_archive_url (str): The direct URL to the .zip archive on Zenodo.
        target_dir (str): The local directory to check for/extract data into (e.g., 'cyber/').

    Returns:
        bool: True if data is ready, False otherwise.
    """
    # Check if the target directory already exists and has .parquet files
    if os.path.exists(target_dir) and any(f.endswith('.parquet') for f in os.listdir(target_dir)):
        print(f"--> Data found locally in '{target_dir}'. Skipping download.")
        return True

    print(f"--> Local data not found. Starting download from Zenodo")
    print(f"    URL: {zenodo_archive_url}")

    try:
        # Download the file content into memory
        response = requests.get(zenodo_archive_url, stream=True)
        response.raise_for_status()

        print("--> Download complete. Unzipping relevant files from archive...")
        
        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Open the zip file from the downloaded bytes
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Iterate through all files in the zip archive
            for member in z.infolist():
                # We only care about files that end in .parquet and are not directories
                if member.filename.endswith('.parquet') and not member.is_dir():
                    # Get just the base filename, ignoring any internal folder structure
                    # e.g., '13787591/Crypto_Desktop.parquet' becomes 'Crypto_Desktop.parquet'
                    base_filename = os.path.basename(member.filename)
                    output_path = os.path.join(target_dir, base_filename)
                    
                    # Extract the file by reading its content and writing to the new path
                    with z.open(member) as source, open(output_path, 'wb') as target:
                        target.write(source.read())
                    
                    print(f"    - Extracted: {base_filename}")
        
        print(f"--> Successfully extracted all .parquet files to '{target_dir}'")
        return True

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download data. Please check the URL and your internet connection. Details: {e}")
        return False
    except zipfile.BadZipFile:
        print("ERROR: Downloaded file is not a valid zip archive.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during data preparation: {e}")
        return False