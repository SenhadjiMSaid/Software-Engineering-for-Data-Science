import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab4_seds.src.models.row_2_list import row_to_list

import pytest

def test_for_clean_row():
    assert row_to_list ("2,081\t314,942\n") == ["2,081","314,942"]

def test_for_missing_area():
    assert row_to_list ("\t314,942\n") is None

def test_for_missing_tab():
    assert row_to_list ("2,081314,942\n") is None
