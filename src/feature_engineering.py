import numpy as np
import pandas as pd
from scipy import stats

def extract_features(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma series de tiempo irregulares (curvas de luz) en 21 descriptores 
    estadísticos, morfológicos y de variabilidad temporal.
    """
    feats = []
    for oid, grp in df_group.groupby('TransientID'):
        g = grp.dropna(subset=['MJD', 'Mag']).sort_values('MJD')
        g = g.drop_duplicates(subset='MJD', keep='first')
        mags, errs, t = g['Mag'].values, g['Magerr'].values, g['MJD'].values
        n = len(mags)
        if n < 4:
            continue

        f = {'TransientID': oid, 'n_points': n}
        f['time_span'] = t[-1] - t[0] if n > 1 else 0.0
        f['mean_mag']  = np.mean(mags)
        f['std_mag']   = np.std(mags, ddof=1) if n > 1 else 0.0
        f['mad_mag']   = stats.median_abs_deviation(mags, scale=1.0)
        f['amplitude'] = mags.max() - mags.min()
        lo, hi = np.percentile(mags, [10, 90])
        f['amp_10_90'] = hi - lo
        for p in (5, 25, 75, 95):
            f[f'pct_{p}'] = np.percentile(mags, p)
        f['skewness'] = stats.skew(mags) if n > 2 else 0.0
        f['kurtosis'] = stats.kurtosis(mags) if n > 3 else 0.0

        # Pendiente lineal
        if n > 1 and f['time_span'] > 0:
            slope, _, r_value, _, _ = stats.linregress(t, mags)
            f['slope'], f['slope_r2'] = slope, r_value ** 2
        else:
            f['slope'], f['slope_r2'] = 0.0, 0.0

        # Índice de Von Neumann
        if n > 2 and f['std_mag'] > 0:
            diffs = np.diff(mags)
            f['vonneumann_eta'] = np.sum(diffs ** 2) / ((n - 1) * np.var(mags, ddof=1))
        else:
            f['vonneumann_eta'] = np.nan

        # Índice J de Stetson
        if n > 2 and np.mean(errs) > 0:
            delta = np.sqrt(n / (n - 1.0)) * (mags - f['mean_mag']) / np.maximum(errs, 1e-6)
            P = delta[:-1] * delta[1:]
            f['stetson_j'] = np.sum(np.sign(P) * np.sqrt(np.abs(P))) / n
        else:
            f['stetson_j'] = 0.0

        # Cruces por la mediana y tiempo al pico
        med = np.median(mags)
        signs = np.sign(mags - med)
        signs = signs[signs != 0]
        f['median_crossings'] = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0

        idx_peak = np.argmin(mags)
        f['time_to_peak_frac'] = (t[idx_peak] - t[0]) / f['time_span'] if f['time_span'] > 0 else 0.5

        # Correlación error-magnitud
        if n > 2 and np.std(errs) > 0 and np.std(mags) > 0:
            f['err_mag_corr'] = np.corrcoef(mags, errs)[0, 1]
        else:
            f['err_mag_corr'] = 0.0
        f['mean_magerr'] = np.mean(errs)

        feats.append(f)
    return pd.DataFrame(feats)