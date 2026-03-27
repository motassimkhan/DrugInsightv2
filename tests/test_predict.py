import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from predict import DDIPredictor
except ImportError:
    pass

@pytest.fixture(scope="module")
def predictor():
    """Initializes the predictor once for the entire test suite."""
    return DDIPredictor()

class TestDDIPredictions:
    def test_known_severe_interaction(self, predictor):
        """
        Test Case 1: Known Severe Interaction.
        Aspirin and Warfarin are known to increase bleeding risk. 
        Expect a high probability, 'Major' severity, and interaction=True.
        """
        result = predictor.predict('Aspirin', 'Warfarin')
        
        assert 'error' not in result
        assert result['interaction'] is True
        assert result['severity'] in ['Major', 'Moderate']
        assert result['probability'] > 0.7
        assert 'risk_index' in result
        assert 'mechanism' in result

        # Validate that the explainer captures the clinical significance
        mechanism_text = result['mechanism'].lower()
        assert 'adverse' in mechanism_text or 'bleeding' in mechanism_text or 'interaction' in mechanism_text

    def test_no_expected_interaction(self, predictor):
        """
        Test Case 2: No Expected Interaction.
        Test a pair with little to no expected overlap to validate Tier 3 (ML fallback).
        Vitamin C (Ascorbic Acid) and Penicillin.
        """
        result = predictor.predict('Ascorbic Acid', 'Penicillin G')
        
        assert 'error' not in result
        # Note: Depending on the GNN threshold and exact data, this might be exactly False or a low probability True.
        # We ensure the probability is low.
        assert result['probability'] < 0.6
        assert result['risk_index'] < 60


class TestErrorHandling:
    def test_invalid_drug_names(self, predictor):
        """
        Test Case 3a: Invalid or unknown drug names.
        """
        result = predictor.predict('FakeDrug123', 'Aspirin')
        assert 'error' in result
        assert 'not found in DrugBank' in result['error']

    def test_identical_drug_inputs(self, predictor):
        """
        Test Case 3b: Identical drug inputs.
        """
        result = predictor.predict('Aspirin', 'aspirin')
        assert 'error' in result
        assert 'refer to the same drug' in result['error']

class TestOutputSchema:
    def test_output_dictionary_schema(self, predictor):
        """
        Test Case 4: Output Schema Validation.
        Ensures the predictor returns the expected dictionary structure for API/UI consumption.
        """
        result = predictor.predict('Ibuprofen', 'Lisinopril')
        
        assert 'error' not in result
        
        expected_keys = [
            'drug_a', 'drug_b', 'drugbank_id_a', 'drugbank_id_b', 
            'interaction', 'probability', 'risk_index', 'severity', 
            'confidence', 'evidence_tier', 'decision_source', 
            'severity_source', 'summary', 'mechanism', 'recommendation', 
            'evidence', 'component_scores', 'uncertainty'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in prediction output"
            
        assert isinstance(result['probability'], float)
        assert isinstance(result['interaction'], bool)
        assert isinstance(result['risk_index'], int)
        assert isinstance(result['evidence'], dict)
        assert 'drugbank' in result['evidence']
        assert 'twosides' in result['evidence']
        assert 'ml' in result['evidence']
