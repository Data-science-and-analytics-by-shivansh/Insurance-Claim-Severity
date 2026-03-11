# pytest.ini - Pytest configuration for Fraud Detection System

[pytest]
# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Paths
testpaths = tests

# Coverage options
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=insurance_claims_analysis
    --cov-report=html
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=75
    --maxfail=5

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    smoke: marks tests as smoke tests for quick validation
    fraud: marks tests specific to fraud detection logic
    quality: marks tests for data quality checking
    performance: marks performance/benchmark tests
    security: marks security-related tests

# Warnings
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning

# Minimum Python version
minversion = 3.8

# Logging
log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(name)s - %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S

# Output
console_output_style = progress

# Timeout (prevent hanging tests)
timeout = 300
timeout_method = thread

# Parallel execution settings (with pytest-xdist)
# Run with: pytest -n auto
# -n auto uses all available CPU cores

# Test order
# --random-order for randomized test execution (requires pytest-random-order)
# --randomly-seed=12345 for reproducible random order
