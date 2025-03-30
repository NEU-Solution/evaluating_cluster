import unittest
import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import modules
sys.path.append('..')

# Import modules to test
from src.scoring import single_scoring_mcq, scoring_mcq, evaluate_generation
from llm import LLM


class MockLLM(LLM):
    """Mock LLM class for testing purposes"""
    def __init__(self, response="\\boxed{A}"):
        self.response = response
        self.model_name = "mock-model"
    
    def __call__(self, messages):
        return self.response


class TestScoring(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_output.jsonl")
        
        # Sample test questions
        self.test_questions = [
            {
                "id": "q1",
                "mcq_question": "What is 1+1?",
                "choice": "A. 2\nB. 3\nC. 4\nD. 5",
                "answer": "A"
            },
            {
                "id": "q2",
                "mcq_question": "What is the capital of France?",
                "choice": "A. London\nB. Berlin\nC. Paris\nD. Rome",
                "answer": "C"
            }
        ]
    
    def tearDown(self):
        """Clean up after each test method"""
        # Remove test output file if it exists
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
        os.rmdir(self.temp_dir)
    
    @patch('src.scoring.extract')
    def test_single_scoring_mcq_correct(self, mock_extract):
        """Test single_scoring_mcq with a correct answer"""
        # Set up mock
        mock_extract.return_value = "A"
        llm = MockLLM("\\boxed{A}")
        
        # Test
        result = single_scoring_mcq(llm, self.test_questions[0], self.output_path)
        
        # Assert
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["id"], "q1")
        
        # Verify file was written
        self.assertTrue(os.path.exists(self.output_path))
    
    @patch('src.scoring.extract')
    def test_single_scoring_mcq_wrong(self, mock_extract):
        """Test single_scoring_mcq with a wrong answer"""
        mock_extract.return_value = "B"
        llm = MockLLM("\\boxed{B}")
        
        result = single_scoring_mcq(llm, self.test_questions[0], self.output_path)
        
        self.assertEqual(result["score"], 0)
    
    @patch('src.scoring.extract')
    def test_single_scoring_mcq_cant_answer(self, mock_extract):
        """Test single_scoring_mcq when answer is 'E' (can't answer)"""
        mock_extract.return_value = "E"
        llm = MockLLM("\\boxed{E}")
        
        result = single_scoring_mcq(llm, self.test_questions[0], self.output_path)
        
        self.assertEqual(result["score"], 0)
    
    @patch('src.scoring.extract')
    def test_single_scoring_mcq_exception(self, mock_extract):
        """Test single_scoring_mcq when an exception occurs"""
        mock_extract.side_effect = Exception("Test exception")
        llm = MockLLM("Invalid response")
        
        result = single_scoring_mcq(llm, self.test_questions[0], self.output_path)
        
        self.assertEqual(result["score"], 0)
    
    @patch('src.scoring.single_scoring_mcq')
    def test_scoring_mcq_single_thread(self, mock_single_scoring):
        """Test scoring_mcq in single thread mode"""
        mock_single_scoring.return_value = {"id": "test", "score": 1}
        llm = MockLLM()
        
        results = scoring_mcq(llm, self.test_questions, self.output_path, multi_thread=False)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_single_scoring.call_count, 2)
    
    @patch('src.scoring.single_scoring_mcq')
    def test_scoring_mcq_multi_thread(self, mock_single_scoring):
        """Test scoring_mcq in multi-thread mode"""
        mock_single_scoring.return_value = {"id": "test", "score": 1}
        llm = MockLLM()
        
        results = scoring_mcq(llm, self.test_questions, self.output_path, max_workers=2, multi_thread=True)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_single_scoring.call_count, 2)
    
    @patch('src.scoring.single_scoring_mcq')
    def test_scoring_mcq_exception_handling(self, mock_single_scoring):
        """Test exception handling in multi-thread mode"""
        # First call succeeds, second call fails
        mock_single_scoring.side_effect = [
            {"id": "q1", "score": 1},
            Exception("Test exception")
        ]
        llm = MockLLM()
        
        results = scoring_mcq(llm, self.test_questions, self.output_path, max_workers=2, multi_thread=True)
        
        # Should still have 2 results, with the second being a failure record
        self.assertEqual(len(results), 2)

        # Check that one of the results has a score of 1
        self.assertTrue(any(result["score"] == 1 for result in results))
    
    @patch('builtins.open')
    @patch('src.scoring.scoring_mcq')
    def test_evaluate_generation(self, mock_scoring_mcq, mock_open):
        """Test evaluate_generation function"""
        # Mock file reading
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = [
            '{"id": "q1", "mcq_question": "test", "choice": "A. test", "answer": "A"}'
        ]
        mock_open.return_value = mock_file
        
        # Mock scoring_mcq function
        mock_scoring_mcq.return_value = [{"id": "q1", "score": 1}]
        
        llm = MockLLM()
        result = evaluate_generation(llm, "test_path.jsonl")
        
        # Verify the function called scoring_mcq with the right parameters
        mock_scoring_mcq.assert_called_once()
        self.assertEqual(result, [{"id": "q1", "score": 1}])


if __name__ == '__main__':
    unittest.main()