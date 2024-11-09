import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lab4_seds.src.models.sentiment import extract_sentiment
import pytest
from textblob import TextBlob
import pytest
import csv

csv_path = "data\soccer_sentiment_analysis.csv"



testdata =[]
with open(csv_path, "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        testdata.append(row)
    

@pytest.mark.parametrize('sample', testdata)
def test_extract_sentiment(sample):

    neg_sentiment = extract_sentiment(sample)

    assert neg_sentiment <= 0