#!/bin/bash

echo "Running comprehensive tests..."

# Activate venv
source venv/bin/activate

# Run unit tests
echo "Running unit tests..."
pytest tests/test_predictions.py -v

# Run API tests
echo "Running API integration tests..."
pytest tests/test_integration.py -v

# Run robustness tests
echo "Running robustness tests..."
pytest tests/test_robustness.py -v

# Run edge case tests
echo "Running edge case tests..."
pytest tests/test_edge_cases.py -v

echo "All tests completed!"