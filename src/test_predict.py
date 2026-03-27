import sys
import os
import json
import pprint

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from predict import DDIPredictor

def main():
    predictor = DDIPredictor()
    # Test pair
    result = predictor.predict("Amoxicillin", "Acetaminophen")
    
    print("\n" + "="*50)
    print("Raw Predictor Output:")
    print("="*50)
    pprint.pprint(result)
    
    print("\n" + "="*50)
    print("Explainer Output:")
    print("="*50)
    
    # `predict()` already invokes Explainer internally and returns explanation fields.
    print(result.get("full_explanation", "No explanation available."))
    
if __name__ == "__main__":
    main()
