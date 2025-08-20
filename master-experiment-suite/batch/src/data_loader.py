"""
Dataset loading helper module for rbd24 datasets.
"""
__docformat__ = 'numpy'
from os import PathLike
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from pathlib import Path
from numpy import unique as npunique, number, array
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas import read_pickle, read_parquet, concat, get_dummies, read_csv, DataFrame
from urllib.request import urlopen
from collections.abc import Callable

_N_TRIES = 5

def _dl_rbd(rbd_dir: Path, data_url: str, lg: Callable) -> None:
    """Helper to download and extract the rbd24 dataset from Zenodo."""
    if not rbd_dir.is_dir():
        lg(f"Downloading rbd24 from {data_url}...")
        try:
            zip_raw = urlopen(data_url).read()
            with ZipFile(BytesIO(zip_raw), 'r') as z:
                rbd_dir.mkdir(parents=True, exist_ok=True)
                lg("Completed download, now extracting zip...")
                z.extractall(rbd_dir)
            lg("rbd24 extracted successfully.")
        except BadZipFile as e:
            raise Exception(f"Failed to download or extract zip file. It may be corrupt. Error: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred during download: {e}")
    else:
        lg(f"Directory '{rbd_dir}' already exists, assuming data is present.")

def _rm_redundant(df: DataFrame, lg: Callable) -> DataFrame:
    """Removes columns from a DataFrame that have only one unique value."""
    redundant_cols = [c for c in df.columns if df[c].nunique() == 1]
    if redundant_cols:
        lg("Redundant columns (all values equal):")
        [lg(f"- {c}") for c in redundant_cols]
        return df.drop(redundant_cols, axis=1)
    return df

def _split_and_scale(x: DataFrame, y: DataFrame, random_state: int, categorical: bool,
                     test_size: float, split_by_user: bool, lg: Callable):
    """Internal function to handle splitting, encoding, and scaling."""
    rng = default_rng(random_state)
    x_processed = x.drop(['user_id', 'timestamp'], axis=1, errors='ignore')

    if categorical:
        x_processed = get_dummies(x_processed)
    else:
        x_processed = x_processed.select_dtypes(include=number)

    x_arr = x_processed.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=bool)
    
    assert len(npunique(y_arr)) > 1, "All labels are the same! Cannot train."

    if split_by_user:
        unique_uids = sorted(x['user_id'].unique())
        rng.shuffle(unique_uids)
        num_test_users = round(len(unique_uids) * test_size)
        test_uids = set(unique_uids[:num_test_users])
        
        test_mask = x['user_id'].isin(test_uids)
        train_mask = ~test_mask
        
        x_train, x_test = x_arr[train_mask], x_arr[test_mask]
        y_train, y_test = y_arr[train_mask], y_arr[test_mask]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_arr, y_arr, test_size=test_size, random_state=rng.integers(2**32), stratify=y_arr
        )

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, y_train, x_test, y_test, sc

def rbd24(rbd_dir: PathLike = Path.home() / 'data' / 'rbd24',
          random_state: int = 1729,
          split_by_user: bool = True,
          categorical: bool = True,
          test_size: float = .2,
          lg: bool = False):
    """
    Loads, caches, and prepares the rbd24 dataset for machine learning.
    
    This function handles downloading, cleaning, one-hot encoding, splitting,
    and scaling the data. It returns dictionaries of train/test sets for each
    dataset category.
    """
    if isinstance(lg, bool):
        lg = print if lg else lambda *args, **kwargs: None
        
    rbd_dir = Path(rbd_dir)
    pickle_file = rbd_dir / 'rbd24_processed.pkl'

    if pickle_file.is_file():
        lg(f"Loading cached DataFrame from '{pickle_file}'...")
        df = read_pickle(pickle_file)
    else:
        data_url = 'https://zenodo.org/api/records/13787591/files-archive'
        _dl_rbd(rbd_dir=rbd_dir, data_url=data_url, lg=lg)
        
        parquet_files = sorted(rbd_dir.glob('*.parquet'))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in '{rbd_dir}'.")
        
        dfs = []
        for f in parquet_files:
            df_cat = read_parquet(f)
            df_cat['category'] = f.stem
            dfs.append(df_cat)
        
        df = concat(dfs, ignore_index=True)
        df = _rm_redundant(df, lg=lg)
        
        lg(f"Saving combined and cleaned DataFrame to '{pickle_file}'...")
        df.to_pickle(pickle_file)

    lg("Final DataFrame columns:", *df.columns)
    x, y = df.drop('label', axis=1), df['label']
    
    X_train, Y_train, X_test, Y_test, scalers = {}, {}, {}, {}, {}
    
    for cat_name in sorted(x['category'].unique()):
        lg(f"\n======== Processing Dataset: {cat_name} ========")
        cat_mask = x['category'] == cat_name
        xc = x[cat_mask].drop('category', axis=1)
        yc = y[cat_mask]
        
        # Free memory by deleting references
        del cat_mask
        
        X_train[cat_name], Y_train[cat_name], X_test[cat_name], Y_test[cat_name], scalers[cat_name] = \
            _split_and_scale(xc, yc, random_state, categorical, test_size, split_by_user, lg)
        
        # Free memory immediately after processing each dataset
        del xc, yc
        lg("==========================================")

    return (X_train, Y_train), (X_test, Y_test), scalers

def summarise_data(xtrn: dict[str, array], ytrn: dict[str, array],
                   xtst: dict[str, array], ytst: dict[str, array]):
    """Summarises split binary classification datasets returned by rbd24."""
    print("Dataset Summary:")
    print("-" * 80)
    for ds in sorted(xtst.keys()):
        print(f"Dataset: {ds:>20} | "
              f"Train: {len(xtrn[ds]):<8} | "
              f"Test: {len(xtst[ds]):<8} | "
              f"Train Positives: {ytrn[ds].mean():.2%} | "
              f"Test Positives: {ytst[ds].mean():.2%}")
    print("-" * 80)

# if __name__ == '__main__':

#     print("Running data loader as a standalone script to download and test...")
#     (X_train, Y_train), (X_test, Y_test), _ = rbd24(lg=print)
#     summarise_data(X_train, Y_train, X_test, Y_test)