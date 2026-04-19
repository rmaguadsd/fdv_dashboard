#!/usr/bin/env python3
import urllib.request
import json
import sys

OLLAMA_BASE = 'http://localhost:11434'

try:
    req = urllib.request.Request(OLLAMA_BASE + '/api/tags')
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read().decode('utf-8'))
    
    print("Full response:")
    print(json.dumps(data, indent=2))
    print("\n" + "="*50)
    print(f"Number of models: {len(data.get('models', []))}")
    print(f"Models: {data.get('models', [])}")
    print("\n" + "="*50)
    
    # Try to extract names different ways
    if 'models' in data:
        models = data['models']
        if models:
            first_model = models[0]
            print(f"First model structure: {first_model}")
            print(f"First model keys: {list(first_model.keys()) if isinstance(first_model, dict) else 'NOT A DICT'}")
            
            # Try different extraction methods
            print("\nDifferent extraction methods:")
            if isinstance(first_model, dict):
                print(f"  m['name']: {first_model.get('name', 'NOT FOUND')}")
                print(f"  m.get('name'): {first_model.get('name', 'NOT FOUND')}")
                print(f"  Keys: {list(first_model.keys())}")
                print(f"  Full: {first_model}")
            else:
                print(f"  First model is not a dict: {first_model}")
        
        names = []
        for m in models:
            if isinstance(m, dict) and 'name' in m:
                names.append(m['name'])
            else:
                print(f"  WARNING: Model doesn't have 'name' key: {m}")
        
        print(f"\nExtracted names: {names}")
        print(f"Sorted names: {sorted(names)}")
        
except Exception as ex:
    print(f"Error: {ex}", file=sys.stderr)
    import traceback
    traceback.print_exc()
