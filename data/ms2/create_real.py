import pickle
from lib.cloud_io import gs_open
from lib.lcms.ms2.libraries import HXMZXML
from tqdm import tqdm

with gs_open('gs://hx-lawrence/outputs/selected_spectra_ids.pickle', 'rb') as f:
    selected_spectra_ids = pickle.load(f)
selected_spectra_ids.sort()

batch_size = 5000
for i in tqdm(range(0, len(selected_spectra_ids), batch_size)):
    ids = selected_spectra_ids[i:i+batch_size]
    spectra = HXMZXML.get_spectra(ids)
    break

data = []
for spectra_id, rows in spectra.groupby("spectra_id"):
    if not (200 < rows.iloc[0].precursor_mz < 1000):
        continue

    data.append([])
    for n, row in rows.iterrows():
        if row.intensity < 1e3:
            continue
            data.append(row["mz"])

data = [d for d in data if len(d) > 0]