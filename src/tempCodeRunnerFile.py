
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