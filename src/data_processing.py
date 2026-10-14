import os
import urllib.request
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/MachineLearningUniandes/MANTRA/master/data/lightcurves/"
MAIN5 = ['SN', 'AGN', 'CV', 'Blazar', 'HPM']

def download_and_load_raw_data(data_dir: str = "../data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Descarga los archivos crudos de MANTRA si no existen localmente y los carga en memoria."""
    os.makedirs(data_dir, exist_ok=True)
    files = {
        'transient_labels.csv': BASE_URL + "transient_labels.csv",
        'transient_lightcurves.csv': BASE_URL + "transient_lightcurves.csv",
        'transient_info.txt': BASE_URL + "transient_info.txt",
    }
    
    paths = {}
    for fname, url in files.items():
        fpath = os.path.join(data_dir, fname)
        paths[fname] = fpath
        if not os.path.exists(fpath):
            print(f"Descargando {fname} ...")
            urllib.request.urlretrieve(url, fpath)
            
    df_labels = pd.read_csv(paths['transient_labels.csv'], dtype={'TransientID': str})
    df_info = pd.read_csv(paths['transient_info.txt'], sep='\t', dtype={'CSS images': str})
    df_lc = pd.read_csv(paths['transient_lightcurves.csv'])
    
    return df_lc, df_labels, df_info


def build_clean_catalog(df_lc: pd.DataFrame, df_labels: pd.DataFrame, df_info: pd.DataFrame, min_obs: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estandariza llaves primarias, filtra ruido de etiquetas y aplica el umbral mínimo de observaciones."""
    df_lc = df_lc.copy()
    df_lc['TransientID'] = df_lc['ID'].str.replace('TranID', '', regex=False)
    
    df_info_clean = df_info.rename(columns={
        'CSS images': 'TransientID',
        'RA (J2000)': 'ra', 
        'Dec (J2000)': 'dec'
    })
    
    # Filtrar solo las 5 clases confirmadas
    df_labels_clean = df_labels[df_labels['Classification'].isin(MAIN5)].copy()
    df_labels_clean = df_labels_clean.rename(columns={'Classification': 'label'})
    
    # Calcular puntos fotométricos por objeto y filtrar por suficiencia estadística
    npts = df_lc.groupby('TransientID').size().rename('n_points')
    df_pool = df_labels_clean.merge(npts, on='TransientID', how='left')
    df_pool['n_points'] = df_pool['n_points'].fillna(0).astype(int)
    df_pool = df_pool[df_pool['n_points'] >= min_obs].copy()
    
    # Integrar coordenadas espaciales
    df_pool = df_pool.merge(df_info_clean[['TransientID', 'ra', 'dec']], on='TransientID', how='left')
    
    return df_pool, df_lc