"""
Pytest Configuration and Shared Fixtures
========================================
Shared test fixtures for Insurance Claims Fraud Detection tests

This file is automatically loaded by pytest
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_output_dir():
    """Create temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_claims_data():
    """Generate sample insurance claims for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'claim_id': [f'CLM-{i:05d}' for i in range(n_samples)],
        'claim_amount': np.random.lognormal(9, 1, n_samples),
        'claim_type': np.random.choice(['Collision', 'Comprehensive', 'Liability', 'Uninsured'], n_samples),
        'vehicle_age': np.random.randint(0, 25, n_samples),
        'driver_age': np.random.randint(18, 85, n_samples),
        'driver_age_group': np.random.choice(['Young', 'Middle', 'Senior'], n_samples),
        'num_claims_history': np.random.poisson(1.5, n_samples),
        'time_to_report': np.random.exponential(10, n_samples),
        'vehicle_value': np.random.lognormal(10, 0.5, n_samples),
        'policy_age': np.random.exponential(5, n_samples),
        'region': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'fraud_flag': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })


@pytest.fixture
def fraud_claims_data():
    """Generate known fraud cases for testing"""
    np.random.seed(42)
    
    return pd.DataFrame({
        'claim_id': [f'FRAUD-{i:03d}' for i in range(50)],
        'claim_amount': np.random.uniform(50000, 150000, 50),  # High amounts
        'claim_type': np.random.choice(['Liability', 'Collision'], 50),
        'vehicle_age': np.random.randint(15, 25, 50),  # Older vehicles
        'driver_age': np.random.randint(20, 40, 50),
        'driver_age_group': ['Young'] * 50,
        'num_claims_history': np.random.randint(5, 15, 50),  # Many prior claims
        'time_to_report': np.random.uniform(0.1, 3, 50),  # Reported very quickly
        'vehicle_value': np.random.uniform(5000, 15000, 50),  # Low value cars
        'policy_age': np.random.uniform(0.1, 2, 50),  # New policies
        'region': np.random.choice(['Urban', 'Suburban'], 50),
        'fraud_flag': [1] * 50
    })


@pytest.fixture
def clean_claims_data():
    """Generate high-quality claims data without issues"""
    np.random.seed(42)
    n_samples = 500
    
    return pd.DataFrame({
        'claim_id': [f'CLM-CLEAN-{i:05d}' for i in range(n_samples)],
        'claim_amount': np.random.lognormal(9, 0.5, n_samples),
        'claim_type': np.random.choice(['Collision', 'Comprehensive'], n_samples),
        'vehicle_age': np.random.randint(0, 15, n_samples),
        'driver_age': np.random.randint(25, 65, n_samples),
        'num_claims_history': np.random.poisson(0.5, n_samples),
        'fraud_flag': [0] * n_samples
    })


@pytest.fixture
def claims_with_quality_issues():
    """Generate claims data with quality problems"""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'claim_id': [f'CLM-{i:05d}' for i in range(n_samples)],
        'claim_amount': [1000.0 if i % 10 != 0 else np.nan for i in range(n_samples)],
        'claim_type': ['Collision' if i % 5 != 0 else None for i in range(n_samples)],
        'vehicle_age': np.random.randint(-5, 30, n_samples),  # Some negative ages
        'driver_age': np.random.randint(10, 100, n_samples),  # Some too young
        'fraud_flag': np.random.choice([0, 1, -1], n_samples)  # Invalid flag values
    })
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]])
    
    return df


@pytest.fixture
def mock_fraud_detector():
    """Pre-configured fraud detector for testing"""
    from insurance_claims_analysis import FraudDetector
    
    return FraudDetector(contamination=0.05)


@pytest.fixture
def trained_fraud_detector(sample_claims_data):
    """Pre-trained fraud detector"""
    from insurance_claims_analysis import FraudDetector
    
    detector = FraudDetector(contamination=0.05)
    detector.fit(sample_claims_data)
    
    return detector


@pytest.fixture
def mock_severity_predictor():
    """Pre-configured severity predictor"""
    from insurance_claims_analysis import SeverityPredictor
    
    from insurance_claims_analysis import SystemConfig
    config = SystemConfig()
    
    return SeverityPredictor(config)


@pytest.fixture
def sample_config():
    """Sample system configuration"""
    from insurance_claims_analysis import SystemConfig
    
    return SystemConfig(
        test_size=0.2,
        random_state=42,
        cv_folds=5
    )


@pytest.fixture
def large_claims_dataset():
    """Generate large dataset for performance testing"""
    np.random.seed(42)
    n_samples = 10000
    
    return pd.DataFrame({
        'claim_amount': np.random.lognormal(9, 1, n_samples),
        'claim_type': np.random.choice(['Collision', 'Comprehensive', 'Liability'], n_samples),
        'vehicle_age': np.random.randint(0, 20, n_samples),
        'driver_age': np.random.randint(18, 80, n_samples),
        'num_claims_history': np.random.poisson(1, n_samples),
        'fraud_flag': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })


# Helper classes
class Helpers:
    """Helper methods for fraud detection tests"""
    
    @staticmethod
    def calculate_fraud_metrics(y_true, y_pred):
        """Calculate fraud detection metrics"""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    @staticmethod
    def is_valid_fraud_score(score):
        """Check if fraud score is valid"""
        return isinstance(score, (int, float)) and 0 <= score <= 1
    
    @staticmethod
    def generate_fraud_pattern(n_samples=100):
        """Generate data with clear fraud patterns"""
        np.random.seed(42)
        
        # Normal claims
        normal = pd.DataFrame({
            'claim_amount': np.random.lognormal(9, 0.5, n_samples),
            'num_claims_history': np.random.poisson(1, n_samples),
            'time_to_report': np.random.exponential(10, n_samples),
            'fraud_flag': [0] * n_samples
        })
        
        # Fraudulent claims (obvious patterns)
        fraud = pd.DataFrame({
            'claim_amount': np.random.uniform(50000, 100000, n_samples // 10),
            'num_claims_history': np.random.randint(5, 10, n_samples // 10),
            'time_to_report': np.random.uniform(0.1, 2, n_samples // 10),
            'fraud_flag': [1] * (n_samples // 10)
        })
        
        return pd.concat([normal, fraud], ignore_index=True)


@pytest.fixture
def helpers():
    """Provide helper methods to tests"""
    return Helpers()


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "fraud: marks tests specific to fraud detection"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "fraud" in item.nodeid:
            item.add_marker(pytest.mark.fraud)
        if "performance" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# Session-level data
@pytest.fixture(scope="session")
def reference_fraud_cases():
    """Reference fraud cases for validation across tests"""
    return {
        'high_amount_quick_report': {
            'claim_amount': 100000,
            'time_to_report': 1.0,
            'num_claims_history': 8,
            'expected_risk': 'HIGH'
        },
        'multiple_claims_old_car': {
            'claim_amount': 30000,
            'vehicle_age': 20,
            'num_claims_history': 10,
            'expected_risk': 'MEDIUM'
        },
        'normal_claim': {
            'claim_amount': 5000,
            'time_to_report': 12,
            'num_claims_history': 1,
            'expected_risk': 'LOW'
        }
    }
