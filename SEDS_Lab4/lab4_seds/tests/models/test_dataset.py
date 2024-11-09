import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import csv
from lab4_seds.src.models.row_2_list import row_to_list
import pytest

# Load your dataset from the CSV file

path = os.path.join('data', 'house_price.csv')
df = pd.read_csv(path, sep=";")
dataset = df.values.tolist()


# Test if the function correctly handles rows with missing values
# Parametrize the test function to iterate through each row in the dataset
@pytest.mark.parametrize("input_row", dataset)
def test_row_to_list_with_missing_values(input_row):
   
    missing_values = any(pd.isna(value) for value in input_row)
    assert not missing_values, f"Missing values found in the row no. {input_row}"

