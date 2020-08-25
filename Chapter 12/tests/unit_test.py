import unittest
import pandas as pd

class TestStringMethods(unittest.TestCase):
  def test_upper(self):
    self.assertEqual('foo'.upper(), 'FOO')


class TestDataFrameStats(unittest.TestCase):
  def setUp(self):
    # initialize and load df
    self.df = pd.DataFrame(data={'data': [0,1,2,3]})

  def test_min(self):
    self.assertEqual(self.df.min().values[0], 0)