# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
from flask import Flask, render_template, jsonify, request, make_response
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import threading
import time
import logging
import uuid
import os
# --- NEW IMPORTS ---
import requests
import zipfile
import io
# --- END NEW IMPORTS ---

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# --- NEW: ZENODO CONFIGURATION ---
ZENODO_ARCHIVE_URL = 'https://zenodo.org/api/records/13787591/files-archive'
DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'data')
# --- END NEW ---

CHUNK_SIZE = 1  # Number of rows to process as a "30-second" chunk


# =============================================================================
# NEW: FEATURE DICTIONARY FOR HUMAN-READABLE INSIGHTS
# =============================================================================
FEATURE_DESCRIPTIONS = {
    # Key DNS & Service Rate Features
    'dst_host_srv_count': 'Rapid DNS Queries (Same Service)',
    'dst_host_srv_rerror_rate': 'Rate of DNS Errors',
    'dst_host_rerror_rate': 'Rate of Connection Errors',
    'srv_rerror_rate': 'Service Error Rate',
    
    # Key HTTP Features
    'http_response_body_len': 'HTTP Response Size',
    'http_response_body_len_ratio': 'HTTP Response Body Length Ratio', # Specific for OGCL insight
    'srv_count': 'Connections to Same Service',
    
    # Traffic Volume Features
    'dst_bytes': 'Destination Bytes Sent',
    'src_bytes': 'Source Bytes Sent',
    
    # General Connection Features
    'dst_host_same_src_port_rate': 'Rate of Same Source Port Connections',
    'protocol_type': 'Protocol Type (e.g., TCP/UDP)',
    'service': 'Network Service (e.g., HTTP, DNS, SMTP)',
    'flag': 'Connection Flag (e.g., SF, REJ)',

    # Default for any other features
    'default': 'Other Network Feature'
}

# Helper function to get a description
def get_feature_description(feature_name):
    return FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace('_', ' ').title())


# =============================================================================
# NEW: DATA DOWNLOAD AND EXTRACTION FUNCTION
# =============================================================================
def download_and_extract_data():
    """
    Checks for the data directory. If not present, downloads and extracts
    the data from the Zenodo archive.
    """
    if os.path.exists(DATA_DIRECTORY) and os.listdir(DATA_DIRECTORY):
        logging.info(f"Data directory '{DATA_DIRECTORY}' already exists and is not empty. Skipping download.")
        return

    logging.info(f"Data directory '{DATA_DIRECTORY}' not found or is empty. Starting download from Zenodo...")
    
    # Ensure the target directory exists
    os.makedirs(DATA_DIRECTORY, exist_ok=True)
    
    zip_file_name = os.path.join(DATA_DIRECTORY, 'data_archive.zip')

    try:
        # 1. Download the archive
        with requests.get(ZENODO_ARCHIVE_URL, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            logging.info(f"Downloading {total_size / (1024*1024):.2f} MB...")
            with open(zip_file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        logging.info("Download complete. Starting extraction...")

        # 2. Extract the archive
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIRECTORY)
        
        logging.info(f"Successfully extracted files to '{DATA_DIRECTORY}'.")

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download data: {e}")
        # Clean up partial download if it exists
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        raise  # Re-raise the exception to stop the app if data is critical
    except zipfile.BadZipFile:
        logging.error("Downloaded file is not a valid zip archive or is corrupted.")
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
        raise
    finally:
        # 3. Clean up the zip file
        if os.path.exists(zip_file_name):
            os.remove(zip_file_name)
            logging.info(f"Cleaned up downloaded archive: {zip_file_name}")

# =============================================================================
# ALGORITHM AND DATA LOADING FUNCTIONS (Unchanged)
# =============================================================================
def OGC(X, y, initial_weights, eta=0.1):
    X_np, y_np, w = np.asarray(X), np.asarray(y), initial_weights
    y_pred = np.zeros(len(X_np))
    for i in range(len(X_np)):
        x, y_actual = X_np[i], y_np[i]
        pred = np.sign(x.dot(w)); y_pred[i] = pred if pred != 0 else 1
        if y_actual * x.dot(w) < 1: w -= eta * (-y_actual * x)
    return w, y_pred

def _ogc_pretrain(X, y, eta=0.1):
    """Single-pass OGC weight initialisation from training data (no UI updates)."""
    w = np.zeros(X.shape[1], dtype=np.float64)
    for i in range(len(X)):
        x_i, y_i = X[i], y[i]
        if y_i * x_i.dot(w) < 1:
            w -= eta * (-y_i * x_i)
    return w


def detect_and_load_dataset(dataset_name, data_dir):
    """Auto-detect dataset type from name and apply the correct loading strategy.

    Returns
    -------
    X_stream : pd.DataFrame   — feature rows to stream through the simulation
    y_stream : pd.Series      — corresponding labels
    pretrained_weights : np.ndarray or None
        Weights pre-trained on the training split (where applicable);
        None for cybersecurity datasets that use prequential evaluation.

    Dataset dispatch (mirrors experiments.py logic):
      UNSW_NB15        → separate _train/_test parquets, StandardScaler, pretrain
      MITBIH_Arrhythmia→ separate _train/_test parquets, pretrain (no scaling needed)
      MNIST            → 80/20 shuffle split, binary label (≥5 → +1), pretrain
      CreditFraud/Kaggle→ temporal 80/20 split, label col 'Class', pretrain
      Cybersecurity    → single file grouped by timestamp, prequential (no pretrain)
    """
    # ── UNSW_NB15 ──────────────────────────────────────────────────────────────
    if 'UNSW' in dataset_name.upper():
        train_path = os.path.join(data_dir, 'UNSW_NB15_train.parquet')
        test_path  = os.path.join(data_dir, 'UNSW_NB15_test.parquet')
        df_tr = pd.read_parquet(train_path).dropna()
        df_te = pd.read_parquet(test_path).dropna()
        feat_cols = [c for c in df_tr.columns if c != 'label']
        X_tr = df_tr[feat_cols].values.astype(np.float64)
        X_te = df_te[feat_cols].values.astype(np.float64)
        y_tr = np.where(df_tr['label'].values == 0, -1, 1)
        y_te = np.where(df_te['label'].values == 0, -1, 1)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        pretrained_weights = _ogc_pretrain(X_tr, y_tr)
        return pd.DataFrame(X_te, columns=feat_cols), pd.Series(y_te), pretrained_weights

    # ── MITBIH Arrhythmia ──────────────────────────────────────────────────────
    if 'MITBIH' in dataset_name.upper():
        train_path = os.path.join(data_dir, 'MITBIH_Arrhythmia_train.parquet')
        test_path  = os.path.join(data_dir, 'MITBIH_Arrhythmia_test.parquet')
        df_tr = pd.read_parquet(train_path)
        df_te = pd.read_parquet(test_path)
        feat_cols = [c for c in df_tr.columns if c != 'label']
        X_tr = df_tr[feat_cols].values.astype(np.float64)
        X_te = df_te[feat_cols].values.astype(np.float64)
        y_tr = np.where(df_tr['label'].values == 0, -1.0, 1.0)
        y_te = np.where(df_te['label'].values == 0, -1.0, 1.0)
        pretrained_weights = _ogc_pretrain(X_tr, y_tr)
        return pd.DataFrame(X_te, columns=feat_cols), pd.Series(y_te), pretrained_weights

    # ── MNIST ──────────────────────────────────────────────────────────────────
    if 'MNIST' in dataset_name.upper():
        fpath = os.path.join(data_dir, f'{dataset_name}.parquet')
        df = pd.read_parquet(fpath)
        df['label_binary'] = np.where(df['label'] >= 5, 1, -1)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(0.8 * len(df))
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        feat_cols = [c for c in df.columns
                     if c not in ['label', 'label_binary']
                     and np.issubdtype(df[c].dtype, np.number)]
        X_tr = train_df[feat_cols].values.astype(np.float64)
        y_tr = train_df['label_binary'].values.astype(np.float64)
        X_te = val_df[feat_cols].values.astype(np.float64)
        y_te = val_df['label_binary'].values.astype(np.float64)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        pretrained_weights = _ogc_pretrain(X_tr, y_tr)
        return pd.DataFrame(X_te, columns=feat_cols), pd.Series(y_te), pretrained_weights

    # ── CreditFraud / Kaggle ───────────────────────────────────────────────────
    if 'KAGGLE' in dataset_name.upper() or 'CREDITFRAUD' in dataset_name.upper():
        fpath = os.path.join(data_dir, f'{dataset_name}.parquet')
        df = pd.read_parquet(fpath)
        if 'user_id' in df.columns:
            df = df.drop(columns=['user_id'])
        if 'Time' in df.columns:
            df = df.sort_values('Time').reset_index(drop=True)
        split_idx = int(0.8 * len(df))
        train_data = df.iloc[:split_idx].copy()
        test_data  = df.iloc[split_idx:].copy()
        label_col  = 'Class' if 'Class' in df.columns else 'label'
        drop_cols  = ['Time', 'timestamp', 'user_id', 'label', 'Class']
        feat_cols  = [c for c in df.columns if c not in drop_cols]
        X_tr = train_data[feat_cols].values.astype(np.float64)
        X_te = test_data[feat_cols].values.astype(np.float64)
        y_tr = np.where(train_data[label_col].values == 0, -1, 1)
        y_te = np.where(test_data[label_col].values == 0, -1, 1)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        pretrained_weights = _ogc_pretrain(X_tr, y_tr)
        return pd.DataFrame(X_te, columns=feat_cols), pd.Series(y_te), pretrained_weights

    # ── Cybersecurity (single file, timestamp grouping, prequential) ───────────
    fpath = os.path.join(data_dir, f'{dataset_name}.parquet')
    df = pd.read_parquet(fpath)
    time_col = ('Time' if 'Time' in df.columns
                else 'timestamp' if 'timestamp' in df.columns
                else None)
    if time_col is None:
        raise ValueError(f"No time column found in {dataset_name}")
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    df = df.sort_values(time_col).reset_index(drop=True)
    x_df = df.groupby(time_col).sum()
    label_col = 'label' if 'label' in x_df.columns else 'Class'
    y_series = x_df[label_col].map(lambda v: 1 if v > 0 else -1)
    x_df = x_df.drop(columns=[label_col])
    return x_df, y_series, None  # prequential — no pre-training

# =============================================================================
# SIMULATION STATE MANAGEMENT
# =============================================================================
class Simulation:
    def __init__(self, dataset_name, X_data, y_data, pretrained_weights=None):
        self.dataset_name, self.X_data, self.y_data = dataset_name, X_data, y_data
        self.status, self.logs, self.alerts = "running", [], []
        self.weights = (pretrained_weights.copy()
                        if pretrained_weights is not None
                        else np.zeros(X_data.shape[1]))
        self.total_positives, self.total_negatives, self.false_positives, self.false_negatives = 0, 0, 0, 0
        self.predicted_threats = 0  # tp + fp: times the model predicted a threat
        self.current_index = 0
        self.is_paused = False
        self.lock = threading.Lock()

    def add_log(self, msg):
        with self.lock:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            
    def add_alert(self, alert):
        with self.lock:
            self.alerts.insert(0, alert)
    
    def pause(self):
        # Simple pause - just set the flag
        self.is_paused = True
    
    def resume(self):
        # Simple resume - just unset the flag
        self.is_paused = False
    
    def toggle_pause(self):
        with self.lock:
            if self.is_paused:
                self.resume()
            else:
                self.pause()
            return self.is_paused
    
    def update_metrics(self, y_true, y_pred):
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
            self.total_positives += (tp + fn)      # actual positives (for FNR)
            self.total_negatives += (tn + fp)      # actual negatives (for FPR)
            self.false_positives += fp
            self.false_negatives += fn
            self.predicted_threats += (tp + fp)    # times model predicted +1
        except ValueError:
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] Metrics Warning: Chunk had only one class.")

    def get_state(self):
        with self.lock:
            fnr = self.false_negatives / self.total_positives if self.total_positives > 0 else 0
            fpr = self.false_positives / self.total_negatives if self.total_negatives > 0 else 0
            progress = self.current_index / len(self.X_data) if len(self.X_data) > 0 else 1
            
            return {
                "dataset_name": self.dataset_name, 
                "status": self.status, 
                "progress": progress, 
                "logs": list(self.logs), 
                "alerts": list(self.alerts), 
                "fnr": float(fnr),
                "fpr": float(fpr),
                "instances_processed": int(self.current_index),
                "total_positives_seen": int(self.predicted_threats),
                "total_negatives_seen": int(self.current_index) - int(self.predicted_threats),
                "is_paused": self.is_paused
            }

# =============================================================================
# GLOBAL STATE and "LIVE" STREAM SIMULATOR (Unchanged)
# =============================================================================
# =============================================================================
# GLOBAL STATE and "LIVE" STREAM SIMULATOR (MODIFIED for new alert text)
# =============================================================================
simulations, simulations_lock = {}, threading.Lock()

def run_live_threat_detection(simulation_id):
    with simulations_lock:
        sim_instance = simulations[simulation_id]

    sim_instance.add_log(f"Starting simulation for dataset: {sim_instance.dataset_name}...")
    total_rows = len(sim_instance.X_data)
    feature_names = sim_instance.X_data.columns

    while sim_instance.current_index < total_rows:
        with sim_instance.lock:
            if sim_instance.status != "running":
                break
            
        start_idx = sim_instance.current_index
        end_idx = sim_instance.current_index + CHUNK_SIZE
        X_chunk = sim_instance.X_data.iloc[start_idx:end_idx]
        y_chunk = sim_instance.y_data.iloc[start_idx:end_idx]
        
        if X_chunk.empty:
            break

        # Always advance the data (simulate continuous data flow)
        sim_instance.current_index = end_idx

        # If paused: skip predictions and metrics, just advance data
        if sim_instance.is_paused:
            chunk_num = (start_idx // CHUNK_SIZE) + 1
            sim_instance.logs.append(f"[{time.strftime('%H:%M:%S')}] PAUSED - Data chunk #{chunk_num} received but no prediction made")
            time.sleep(2)  # Continue at normal pace
            continue

        # When not paused: run predictions and update metrics
        current_weights = sim_instance.weights
        new_weights, y_pred = OGC(X_chunk, y_chunk.to_numpy(), current_weights)

        with sim_instance.lock:
            chunk_num = (sim_instance.current_index // CHUNK_SIZE) + 1
            sim_instance.logs.append(f"[{time.strftime('%H:%M:%S')}] ACTIVE - Processing chunk #{chunk_num} with prediction...")
            
            sim_instance.weights = new_weights
            sim_instance.update_metrics(y_chunk, y_pred)
            
            threat_indices = np.where(y_pred == 1)[0]
            if len(threat_indices) > 0:
                threat_idx_in_chunk = threat_indices[0]
                traffic_instance = X_chunk.iloc[threat_idx_in_chunk]
                contributions = traffic_instance.values * current_weights
                
                # --- NEW: NORMALIZE CONTRIBUTIONS TO PERCENTAGES ---
                positive_contributions = contributions[contributions > 0]
                total_positive_score = positive_contributions.sum()
                
                # Avoid division by zero if the total score is zero or negative
                if total_positive_score <= 0:
                    total_positive_score = 1 
                
                percentages = (contributions / total_positive_score) * 100
                # --- END NEW ---

                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'contribution': contributions,
                    'percentage': percentages # Add the new percentage column
                }).sort_values(by='contribution', ascending=False)
                
                top_features = feature_importance_df.head(3).to_dict('records')
                
                is_true_threat = (y_chunk.iloc[threat_idx_in_chunk] == 1)
                forecast_text = "Malicious" if is_true_threat else "Benign"

                alert = {
                    "id": str(uuid.uuid4()),
                    "timestamp": time.strftime('%H:%M:%S'),
                    "forecast": forecast_text,
                    "top_features": top_features
                }
                
                sim_instance.alerts.insert(0, alert)
                # Updated log to show percentage
                top_features_str = ', '.join([f"{get_feature_description(f['feature'])} ({f['percentage']:.1f}%)" for f in top_features])
                sim_instance.logs.append(f"[{time.strftime('%H:%M:%S')}] THREAT FORECASTED: {forecast_text}. Top contributors: {top_features_str}")
            
            sim_instance.current_index = end_idx

        time.sleep(2)

    with sim_instance.lock:
        sim_instance.status = "complete"
        sim_instance.add_log("Simulation finished.")
    
    # Clean up the simulation from global dictionary after a longer delay
    def cleanup_simulation():
        time.sleep(30)  # Wait 30 seconds to allow final status checks
        with simulations_lock:
            if simulation_id in simulations:
                sim = simulations[simulation_id]
                if sim.status in ["complete", "stopped"]:
                    del simulations[simulation_id]
                    logging.info(f"Cleaned up completed simulation {simulation_id}")
    
    cleanup_thread = threading.Thread(target=cleanup_simulation)
    cleanup_thread.daemon = True
    cleanup_thread.start()


# =============================================================================
# FLASK APPLICATION ROUTES (Unchanged)
# =============================================================================
app = Flask(__name__)

def find_available_datasets():
    """Return canonical dataset names for all parquets found in DATA_DIRECTORY.

    Paired files (e.g. UNSW_NB15_train / UNSW_NB15_test) are collapsed into a
    single canonical entry (e.g. UNSW_NB15).
    """
    if not os.path.exists(DATA_DIRECTORY):
        logging.error(f"Data directory '{DATA_DIRECTORY}' not found!")
        return []
    try:
        datasets = set()
        for f in os.listdir(DATA_DIRECTORY):
            if not f.endswith('.parquet'):
                continue
            name = f.replace('.parquet', '')
            # Collapse _train / _test pairs into their canonical name
            if name.endswith('_train') or name.endswith('_test'):
                canonical = name.rsplit('_', 1)[0]
                datasets.add(canonical)
            else:
                datasets.add(name)
        return sorted(datasets)
    except Exception as e:
        logging.error(f"Error scanning data directory: {e}")
        return []

@app.route('/')
def index():
    available_datasets = find_available_datasets()
    # Pass the descriptions dictionary to the template
    return render_template('index.html', datasets=available_datasets, descriptions=FEATURE_DESCRIPTIONS)

@app.route('/start-simulation', methods=['POST'])
def start_simulation():
    data = request.get_json()
    dataset_name = data.get('dataset')
    if not dataset_name:
        return jsonify({"status": "error", "message": "No dataset selected."}), 400

    try:
        X_data, y_data, pretrained_weights = detect_and_load_dataset(dataset_name, DATA_DIRECTORY)
    except Exception as e:
        logging.error(f"Failed to load {dataset_name}: {e}")
        return jsonify({"status": "error", "message": f"Failed to load data for {dataset_name}: {e}"}), 400

    simulation_id = str(uuid.uuid4())
    sim_instance = Simulation(dataset_name, X_data, y_data, pretrained_weights)
    
    with simulations_lock:
        simulations[simulation_id] = sim_instance
    
    thread = threading.Thread(target=run_live_threat_detection, args=(simulation_id,))
    thread.start()
    return jsonify({"status": "success", "simulation_id": simulation_id})

@app.route('/stop-simulation', methods=['POST'])
def stop_simulation():
    data = request.get_json()
    simulation_id = data.get('simulation_id')
    with simulations_lock:
        if simulation_id in simulations:
            sim_instance = simulations[simulation_id]
            with sim_instance.lock:
                if sim_instance.status == "running":
                    sim_instance.status = "stopped"
                    sim_instance.logs.append(f"[{time.strftime('%H:%M:%S')}] Simulation stopped by user.")
            
            # Don't immediately cleanup - let the main cleanup thread handle it
    
    return jsonify({"status": "success"})

@app.route('/pause-simulation', methods=['POST'])
def pause_simulation():
    logging.info("Pause simulation endpoint called")
    data = request.get_json()
    logging.info(f"Request data: {data}")
    simulation_id = data.get('simulation_id')
    logging.info(f"Simulation ID: {simulation_id}")
    
    # First, find the simulation without holding locks for too long
    sim_instance = None
    try:
        with simulations_lock:
            logging.info(f"Acquiring simulations_lock...")
            logging.info(f"Active simulations: {list(simulations.keys())}")
            if simulation_id in simulations:
                sim_instance = simulations[simulation_id]
                logging.info(f"Simulation found in dictionary")
            else:
                logging.error(f"Simulation {simulation_id} not found")
                return jsonify({"status": "error", "message": "Simulation not found"}), 404
        
        logging.info(f"Released simulations_lock, working with simulation...")
        
        # Now work with the simulation instance outside the simulations_lock
        logging.info(f"Found simulation. Status: {sim_instance.status}, Is paused: {sim_instance.is_paused}")
        
        # Always allow toggling pause, regardless of current status
        current_pause_state = sim_instance.is_paused
        logging.info(f"Current pause state: {current_pause_state}")
        
        if current_pause_state:
            logging.info("Attempting to resume...")
            sim_instance.resume()
            logging.info("Resume completed")
        else:
            logging.info("Attempting to pause...")
            sim_instance.pause()
            logging.info("Pause completed")
        
        new_pause_state = sim_instance.is_paused
        logging.info(f"New pause state: {new_pause_state}")
        
        return jsonify({
            "status": "success", 
            "is_paused": new_pause_state,
            "message": "paused" if new_pause_state else "resumed"
        })
        
    except Exception as e:
        logging.error(f"Error in pause_simulation: {e}")
        return jsonify({"status": "error", "message": f"Internal error: {e}"}), 500

@app.route('/simulation-status')
def simulation_status():
    simulation_id = request.args.get('id')
    with simulations_lock:
        if simulation_id in simulations:
            return jsonify(simulations[simulation_id].get_state())
    
    # If simulation not found, return a "completed" status to stop polling
    logging.warning(f"Simulation {simulation_id} not found - returning completed status")
    return jsonify({
        "status": "complete", 
        "message": "Simulation not found - likely completed or stopped",
        "progress": 1.0,
        "logs": [],
        "alerts": [],
        "fnr": 0.0,
        "fpr": 0.0,
        "instances_processed": 0,
        "total_positives_seen": 0,
        "total_negatives_seen": 0,
        "is_paused": False
    })

@app.route('/debug-simulations')
def debug_simulations():
    """Debug endpoint to see active simulations"""
    with simulations_lock:
        debug_info = {
            "active_simulations": len(simulations),
            "simulation_ids": list(simulations.keys()),
            "simulation_states": {
                sim_id: {
                    "status": sim.status,
                    "is_paused": sim.is_paused,
                    "current_index": sim.current_index,
                    "total_rows": len(sim.X_data)
                }
                for sim_id, sim in simulations.items()
            }
        }
    return jsonify(debug_info)

# =============================================================================
# SCRIPT ENTRY POINT (MODIFIED)
# =============================================================================
if __name__ == '__main__':
    # --- NEW: RUN THE DATA DOWNLOADER ON STARTUP ---
    try:
        download_and_extract_data()
    except Exception as e:
        logging.critical(f"CRITICAL: Could not download or process initial dataset. Exiting. Error: {e}")
        exit() # Stop the app if data can't be loaded
    # --- END NEW ---
    
    app.run(debug=True)
