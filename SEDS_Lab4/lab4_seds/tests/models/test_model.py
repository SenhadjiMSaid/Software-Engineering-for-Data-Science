import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab4_seds.src.models.module import serve_beer

import pytest

def test_serve_beer_legal():
    adult = 25
    assert serve_beer(adult) == "Have beer"

def test_serve_beer_illegal():
    child = 10
    assert serve_beer(child) == "No beer"