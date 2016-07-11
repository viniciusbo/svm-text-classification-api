import unittest
from textclf import classificate

class TestTextClf(unittest.TestCase):
  def test_classificate_task(self):
    res = classificate.apply_async(args=['acho que estou gripado'])
    pred = res.wait(timeout=None, propagate=True, interval=0.5)
    self.assertEqual(pred, 4)

if __name__ == '__main__':
    unittest.main()