#!/usr/bin/env python3
"""Test external MongoDB Atlas connectivity"""

import os
import fiftyone as fo
from dotenv import load_dotenv

print('=== Testing External MongoDB Atlas ===')

# Load credentials
load_dotenv()
db_password = os.getenv("MONGODB_PASSWORD")
print('Password loaded:', '✅' if db_password else '❌')

# Configure external MongoDB
print('Configuring external MongoDB Atlas...')
os.environ['FIFTYONE_DATABASE_URI'] = f"mongodb+srv://rithviggolf:{db_password}@roboticdata.pqtfhwu.mongodb.net/"
os.environ['FIFTYONE_DATABASE_NAME'] = "fiftyone_test"

print('FiftyOne version:', fo.__version__)
print('Database URI configured:', os.environ.get('FIFTYONE_DATABASE_URI', 'None')[:50] + '...')

try:
    print('\nAttempting to list datasets from Atlas...')
    datasets = fo.list_datasets()
    print('✅ External MongoDB Atlas working!')
    print('Available datasets:', datasets)
    
    # Test creating a simple dataset
    print('\nTesting dataset creation on Atlas...')
    test_name = "test_external"
    if test_name in datasets:
        test_dataset = fo.load_dataset(test_name)
        print(f'Loaded existing dataset: {test_name}')
    else:
        test_dataset = fo.Dataset(test_name)
        print(f'Created new dataset: {test_name}')
    
    print(f'Dataset has {len(test_dataset)} samples')
    print('✅ External MongoDB Atlas fully working!')
    
except Exception as e:
    print('❌ External MongoDB Atlas failed!')
    print('Error type:', type(e).__name__)
    print('Error message:', str(e)[:300])
    import traceback
    traceback.print_exc()