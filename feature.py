import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from encoder import FragmentEncoder
from spectrum import preprocess
import scipy.sparse as ss

from ms_io import ms_io

logger = logging.getLogger('MS2VEC')


def _peaks_to_features(filename: str,
                       metadata: Optional[pd.DataFrame],
                       spectrum_preprocessing: Dict[str, Any],
                       enc: FragmentEncoder)\
        -> Tuple[str, Optional[pd.DataFrame], Optional[List[ss.csr_matrix]]]:
    """
    Convert the spectra with the given identifiers in the given file to a
    feature array.
    """
    if not os.path.isfile(filename):
        logger.warning('Missing peak file %s, no features generated', filename)
        return filename, None, None
    logger.debug('Process file %s', filename)
    file_scans, file_mz, file_charge, file_encodings = [], [], [], []
    if metadata is not None:
        metadata = metadata.reset_index(['filename'], drop=True)
    for spec in ms_io.get_spectra(filename):
        # noinspection PyUnresolvedReferences
        if ((metadata is None or np.int64(spec.identifier) in metadata.index)
                and preprocess(
                    spec, **spectrum_preprocessing).is_valid):
            file_scans.append(spec.identifier)
            file_mz.append(spec.precursor_mz)
            file_charge.append(spec.precursor_charge)
            file_encodings.append(enc.encode(spec))
    scans = pd.DataFrame({'scan': file_scans, 'charge': file_charge,
                          'mz': file_mz})
    scans['scan'] = scans['scan'].astype(np.int64)
    return filename, scans, file_encodings
