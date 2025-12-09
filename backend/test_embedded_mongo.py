#!/usr/bin/env python3
"""Test embedded MongoDB connectivity"""

import fiftyone as fo

print('=== Testing Embedded MongoDB ===')
print('FiftyOne version:', fo.__version__)

try:
    print('Attempting to list datasets...')
    datasets = fo.list_datasets()
    print('✅ Embedded MongoDB working!')
    print('Available datasets:', datasets)
    
    # Test creating a simple dataset
    print('\nTesting dataset creation...')
    test_name = "test_embedded"
    if test_name in datasets:
        test_dataset = fo.load_dataset(test_name)
        print(f'Loaded existing dataset: {test_name}')
    else:
        test_dataset = fo.Dataset(test_name)
        print(f'Created new dataset: {test_name}')
    
    print(f'Dataset has {len(test_dataset)} samples')
    print('✅ Embedded MongoDB fully working!')
    
except Exception as e:
    print('❌ Embedded MongoDB failed!')
    print('Error type:', type(e).__name__)
    print('Error message:', str(e)[:300])
    import traceback
    traceback.print_exc()