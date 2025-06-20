"""Some module tests."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from math import isclose
from pathlib import Path

import importlib_resources
import numpy as np
import pandas as pd

from analytical_fm.utils import calc_sampling_metrics, clean_sample


def test_clean_sample() -> None:
    """Function to clean_sample."""

    samples_to_clean = ['<bos> C C ( C ) O C ( = O ) c 1 c n c c ( N c 2 c c ( Cl ) c c ( O C C C O S ( C ) ( = O ) = O ) c 2 Cl ) c 1 <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>',
                        '<bos> C C S ( = O ) ( = O ) N c 1 c c ( Cl ) c c ( O c 2 c c c c ( N C 3 = C ( C ) C ( = O ) O C 3 ) c 2 ) c 1 O C C Cl <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>',
                        '<bos> C C S ( = O ) ( = O ) N c 1 c c ( Cl ) c c ( O c 2 c c c c ( N C 3 = C ( C ) C O C 3 = O ) c 2 ) c 1 O C C Cl <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>',
                        '<bos> C C S ( = O ) ( = O ) N c 1 c c ( Cl ) c c ( O c 2 c c c c ( N C 3 = C ( C ) C ( = O ) N C 3 = O ) c 2 ) c 1 O C C O <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>',
                        '<bos> C C S ( = O ) ( = O ) N c 1 c c ( Cl ) c c ( O c 2 c c c c 3 c 2 C ( C O C ( = O ) C Cl ) = C ( C ) N C 3 = O ) c 1 <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>']
    
    cleaned_samples_truth = np.array(['CC(C)OC(=O)c1cncc(Nc2cc(Cl)cc(OCCCOS(C)(=O)=O)c2Cl)c1',
                                'CCS(=O)(=O)Nc1cc(Cl)cc(Oc2cccc(NC3=C(C)C(=O)OC3)c2)c1OCCCl',
                                'CCS(=O)(=O)Nc1cc(Cl)cc(Oc2cccc(NC3=C(C)COC3=O)c2)c1OCCCl',
                                'CCS(=O)(=O)Nc1cc(Cl)cc(Oc2cccc(NC3=C(C)C(=O)NC3=O)c2)c1OCCO',
                                'CCS(=O)(=O)Nc1cc(Cl)cc(Oc2cccc3c(=O)[nH]c(C)c(COC(=O)CCl)c23)c1'])

    cleaned_samples = np.array([clean_sample(sample, True) for sample in samples_to_clean])

    assert np.all(cleaned_samples_truth == cleaned_samples)


def test_scoring() -> None:
    """Function to scoring."""

    with importlib_resources.files("analytical_fm") as source_folder:
        test_data_path = Path(source_folder / "resources/test_data/scoring/test_data.pkl")
        test_data = pd.read_pickle(test_data_path)

        metrics = calc_sampling_metrics(test_data['predictions'], test_data['targets'])

        assert isclose(metrics['Top-1'], 0.2) & isclose(metrics['Top-10'], 0.6)

