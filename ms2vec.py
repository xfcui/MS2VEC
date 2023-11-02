import os
import math
import numpy as np
import pandas as pd
import click
import tempfile
import joblib
from feature import _peaks_to_features
from encoder import FragmentEncoder
from typing import List
import scipy.sparse as ss
import config
from model import MS2VEC
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


def cli_embed(peak_in: List[str], embed_name: str) -> None:
    """
    Embed spectra.

    Convert MS/MS spectra in the PEAK_IN peak files to 128-dimensional
    embeddings using the MS2VEC.

    Supported formats for peak files in PEAK_IN are: mzML, mzXML, MGF.

    """
    if len(peak_in) == 0:
        raise click.BadParameter('No input peak files specified')

    peak_filenames_chunked = np.array_split(peak_in, math.ceil(len(peak_in) / 200))
    temp_dir = tempfile.mkdtemp()
    embed_dir = os.path.join(temp_dir, 'embed')
    os.mkdir(embed_dir)

    preprocessing = {
        'mz_min': config.fragment_mz_min,
        'mz_max': config.fragment_mz_max,
        'min_peaks': config.min_peaks,
        'min_mz_range': config.min_mz_range,
        'remove_precursor_tolerance': config.remove_precursor_tolerance,
        'min_intensity': config.min_intensity,
        'max_peaks_used': config.max_peaks_used,
        'scaling': config.scaling
    }
    fragment_encoding = {
        'min_mz': config.fragment_mz_min,
        'max_mz': config.fragment_mz_max,
        'bin_size': config.bin_size
    }
    charges = config.charges

    enc = FragmentEncoder(**fragment_encoding)
    batch_size = 256

    ms2vec = MS2VEC()
    chkp = torch.load('Data/model_param.pt')
    chkp = {k.replace('module.child_model.', ''): v for k, v in chkp.items()}
    ms2vec.load_state_dict(chkp)
    ms2vec = ms2vec.cuda()

    filename_embeddings = os.path.join(embed_dir, 'MSVEC.npy')

    scans = []
    for i, chunk_filenames in enumerate(peak_filenames_chunked):
        encodings = []
        for filename, file_scans, file_encodings in joblib.Parallel(n_jobs=-1)(
                joblib.delayed(_peaks_to_features)
                (filename, None, preprocessing, enc) for filename in chunk_filenames):
            if file_scans is not None:
                if charges is not None:
                    file_scans = file_scans[
                        (file_scans['charge'] >= charges[0]) &
                        (file_scans['charge'] <= charges[1])].copy()
                if len(file_scans) > 0:
                    file_scans['filename'] = filename
                    scans.append(file_scans)
                    encodings.extend(np.asarray(file_encodings)[file_scans.index.values])  # ?
                    # encodings.extend(file_encodings)
        if len(encodings) > 0:
            _embed_and_save(encodings, batch_size, ms2vec, filename_embeddings.replace('.npy', f'_{i}.npy'))  # 与model相关 加载模型 输出embedding

    if len(scans) > 0:
        meta_name = f'{os.path.splitext(embed_name)[0]}.parquet'
        scans = pd.concat(scans, ignore_index=True, sort=False, copy=False)
        scans[['filename', 'scan', 'charge', 'mz']].to_parquet(
            meta_name, index=False)
        # Merge all temporary embeddings into a single file.
        embeddings = [np.load(filename_embeddings.replace(
                '.npy', f'_{i}.npy'), mmap_mode='r')
                for i in range(len(peak_filenames_chunked))]
        np.save(embed_name, np.vstack(embeddings))
        for i in range(len(peak_filenames_chunked)):
            os.remove(filename_embeddings.replace('.npy', f'_{i}.npy'))


class encoding_ds(Dataset):

    def __init__(self, encodings) -> None:
        super().__init__()
        self.encodings = ss.vstack(encodings, 'csr').toarray()

    def __getitem__(self, index):
        return self.encodings[index]

    def __len__(self):
        return len(self.encodings)


def get_embeddings(dl, model):
    embeddings = []
    model.eval()
    for data in tqdm(dl):
        data = data.to(torch.float32).cuda()
        with torch.no_grad():
            embedding, _ = model(data)
        embeddings.append(embedding.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _embed_and_save(encodings: List[ss.csr_matrix], batch_size: int, ms2vec,
                    filename: str) -> None:
    """
    Embed the given encodings and save them as a NumPy file.
    """
    ds = encoding_ds(encodings)
    dl = DataLoader(ds, batch_size, shuffle=False)

    embeddings_save = get_embeddings(dl, ms2vec)
    np.save(filename, embeddings_save)
