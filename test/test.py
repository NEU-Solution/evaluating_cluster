import unittest
import sys
import os

# Add parent directory to path
sys.path.append('..')

# Import all test modules
from test.test_scoring import TestScoring

# Add more test classes as you create them
# from test_other_module import TestOtherModule

if __name__ == '__main__':
    # Run all tests
    unittest.main()