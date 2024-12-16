"""Test the data analysis utils."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

from pathlib import Path

from mmbart.data_analysis.utils import (
    df_to_hf_dataset,
    exclude_unsupported_files,
    file_to_hf_dataset,
    file_to_pandas_df,
    list_files,
)


def test_list_files():
    """Test folder exploration and file list creation.
    """
    data_path = Path('tests/test_data')
    files_list = list_files(data_path)
    assert(len(files_list)==3)
    
def test_filter_unsupported_files():
    """Test filtering of unsupported files.
    """
    data_path = Path('tests/test_data')
    files_list = list_files(data_path)
    filtered_list = exclude_unsupported_files(files_list, '.csv')
    assert(len(filtered_list)==2)

def test_file_to_pandas_df():
    """Tests conversion of csv to pandas.
    """
    data_path = Path('tests/test_data')
    files_list = list_files(data_path)
    filtered_list = exclude_unsupported_files(files_list, '.csv')
    assert(len(filtered_list)==2)
    data_frames=[]
    for file_path in filtered_list:
        data_frames.append(file_to_pandas_df(file_path))
    assert(len(data_frames)==2)
    assert(len(data_frames[0].columns)==3)

def test_file_to_hf_dataset():
    """test conversion of csv to HF dataset dict.
    """
    data_path = Path('tests/test_data')
    files_list = list_files(data_path)
    filtered_list = exclude_unsupported_files(files_list, '.csv')
    assert(len(filtered_list)==2)
    datasets=[]
    for file_path in filtered_list:
        datasets.append(file_to_hf_dataset(file_path))
    assert(len(datasets)==2)

def test_df_to_hf_dataset():
    """Tests dataframe to HF dataset dic.
    """
    data_path = Path('tests/test_data')
    files_list = list_files(data_path)
    filtered_list = exclude_unsupported_files(files_list, '.csv')
    assert(len(filtered_list)==2)
    data_frames=[]
    for file_path in filtered_list[:1]:
        data_frames.append(file_to_pandas_df(file_path))
    
    dataset_dict = df_to_hf_dataset(data_frames[0], metadata={"description": "this is a test."})
    assert dataset_dict["train"].info.description == "this is a test."
