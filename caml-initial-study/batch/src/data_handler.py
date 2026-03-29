"""
Data Handler for RBD24 Cybersecurity Dataset Downloads
Handles downloading and extracting data from Zenodo repository
"""

import requests
import zipfile
import os
from pathlib import Path
import tempfile
import shutil


def prepare_data_from_zenodo(zenodo_url, target_path, max_retries=3):
    """
    Download and extract RBD24 dataset from Zenodo
    
    Args:
        zenodo_url (str): Zenodo API URL for the dataset
        target_path (str): Local path where data should be extracted
        max_retries (int): Maximum number of download attempts
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    target_dir = Path(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading RBD24 dataset from Zenodo...")
    print(f"URL: {zenodo_url}")
    print(f"Target: {target_dir}")
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            
            # Download the archive
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / "rbd24_data.zip"
                
                print("Downloading archive...")
                response = requests.get(zenodo_url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Write to temporary file with progress indication
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
                
                print("\nExtracting archive...")
                
                # Extract the archive
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find and move parquet files
                extracted_files = list(Path(temp_dir).glob("**/*.parquet"))
                
                if not extracted_files:
                    print("WARNING: No parquet files found in downloaded archive")
                    continue
                
                print(f"Found {len(extracted_files)} parquet files")
                
                # Move parquet files to target directory
                for parquet_file in extracted_files:
                    target_file = target_dir / parquet_file.name
                    shutil.move(str(parquet_file), str(target_file))
                    print(f"Moved: {parquet_file.name}")
                
                print(f"Successfully prepared {len(extracted_files)} dataset files")
                return True
                
        except requests.RequestException as e:
            print(f"Network error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Download failed.")
                return False
                
        except zipfile.BadZipFile as e:
            print(f"Archive corruption error: {e}")
            return False
            
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Preparation failed.")
                return False
    
    return False


def verify_data_integrity(data_path):
    """
    Verify that downloaded data is complete and valid
    
    Args:
        data_path (str): Path to the data directory
        
    Returns:
        dict: Verification results with file counts and status
    """
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        return {'status': 'error', 'message': 'Data directory does not exist'}
    
    parquet_files = list(data_dir.glob('*.parquet'))
    
    if not parquet_files:
        return {'status': 'error', 'message': 'No parquet files found'}
    
    # Expected RBD24 dataset files
    expected_files = [
        'Crypto_desktop.parquet',
        'Crypto_smartphone.parquet',
        'NonEnc_desktop.parquet',
        'NonEnc_smartphone.parquet',
        'OutFlash_desktop.parquet',
        'OutFlash_smartphone.parquet',
        'OutTLS_desktop.parquet',
        'OutTLS_smartphone.parquet'
    ]
    
    found_files = [f.name for f in parquet_files]
    missing_files = [f for f in expected_files if f not in found_files]
    
    verification_results = {
        'status': 'success' if not missing_files else 'partial',
        'total_files': len(parquet_files),
        'expected_files': len(expected_files),
        'found_files': found_files,
        'missing_files': missing_files
    }
    
    if missing_files:
        verification_results['message'] = f"Missing {len(missing_files)} expected files"
    else:
        verification_results['message'] = 'All expected files present'
    
    return verification_results


def get_dataset_info(data_path):
    """
    Get information about available datasets
    
    Args:
        data_path (str): Path to the data directory
        
    Returns:
        dict: Information about each dataset file
    """
    
    import pandas as pd
    
    data_dir = Path(data_path)
    dataset_info = {}
    
    if not data_dir.exists():
        return dataset_info
    
    parquet_files = list(data_dir.glob('*.parquet'))
    
    for file_path in parquet_files:
        try:
            # Read just the first few rows to get basic info
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            dataset_name = file_path.stem
            dataset_info[dataset_name] = {
                'file_path': str(file_path),
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'columns': list(df.columns),
                'has_user_id': 'user_id' in df.columns,
                'has_label': 'label' in df.columns,
                'unique_labels': df['label'].unique().tolist() if 'label' in df.columns else [],
                'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else 0
            }
            
        except Exception as e:
            dataset_info[file_path.stem] = {
                'file_path': str(file_path),
                'error': f"Failed to read file: {e}"
            }
    
    return dataset_info


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "download":
            # Download data
            zenodo_url = 'https://zenodo.org/api/records/13787591/files-archive'
            target_path = '../cyber/'
            
            success = prepare_data_from_zenodo(zenodo_url, target_path)
            
            if success:
                print("Download completed successfully!")
                
                # Verify data
                verification = verify_data_integrity(target_path)
                print(f"Verification: {verification}")
                
                # Show dataset info
                info = get_dataset_info(target_path)
                print(f"\nAvailable datasets: {len(info)}")
                for name, details in info.items():
                    if 'error' not in details:
                        print(f"  {name}: {details['num_rows']} rows, {details['num_columns']} cols")
            else:
                print("Download failed!")
                sys.exit(1)
                
        elif sys.argv[1] == "verify":
            # Verify existing data
            target_path = '../cyber/'
            verification = verify_data_integrity(target_path)
            print(f"Verification results: {verification}")
            
        elif sys.argv[1] == "info":
            # Show dataset information
            target_path = '../cyber/'
            info = get_dataset_info(target_path)
            
            print(f"Dataset Information:")
            print("=" * 50)
            
            for name, details in info.items():
                print(f"\n{name}:")
                if 'error' in details:
                    print(f"  Error: {details['error']}")
                else:
                    print(f"  Rows: {details['num_rows']:,}")
                    print(f"  Columns: {details['num_columns']}")
                    print(f"  Size: {details['file_size_mb']:.1f} MB")
                    print(f"  Users: {details['unique_users']:,}")
                    print(f"  Labels: {details['unique_labels']}")
    else:
        print("Usage:")
        print("  python data_handler.py download  # Download RBD24 data")
        print("  python data_handler.py verify   # Verify data integrity")
        print("  python data_handler.py info     # Show dataset information")
