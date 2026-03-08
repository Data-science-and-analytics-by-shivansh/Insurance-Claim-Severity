"""
Unit Tests for Data Quality Checker Module
==========================================
Tests for insurance_claims_analysis.py DataQualityChecker class

Run with: pytest tests/test_data_quality.py -v
"""

import pytest
import pandas as pd
import numpy as np
from insurance_claims_analysis import (
    DataQualityChecker
)


class TestDataQualityChecker:
    """Test suite for DataQualityChecker class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample insurance claims data"""
        np.random.seed(42)
        return pd.DataFrame({
            'claim_id': [f'CLM-{i:05d}' for i in range(100)],
            'claim_amount': np.random.uniform(1000, 50000, 100),
            'claim_type': np.random.choice(['Collision', 'Comprehensive', 'Liability'], 100),
            'vehicle_age': np.random.randint(0, 20, 100),
            'driver_age': np.random.randint(18, 80, 100),
            'num_claims_history': np.random.randint(0, 10, 100),
            'fraud_flag': np.random.choice([0, 1], 100, p=[0.95, 0.05])
        })
    
    @pytest.fixture
    def data_with_nulls(self):
        """Create data with missing values"""
        df = pd.DataFrame({
            'claim_id': [f'CLM-{i:05d}' for i in range(100)],
            'claim_amount': [1000.0 if i % 10 != 0 else np.nan for i in range(100)],
            'claim_type': ['Collision' if i % 5 != 0 else None for i in range(100)],
            'vehicle_age': np.random.randint(0, 20, 100)
        })
        return df
    
    def test_check_missing_values(self, data_with_nulls):
        """Test missing value detection"""
        checker = DataQualityChecker()
        
        result = checker.check_missing_values(data_with_nulls)
        
        assert isinstance(result, dict)
        assert 'claim_amount' in result
        assert 'claim_type' in result
        assert result['claim_amount'] > 0
        assert result['claim_type'] > 0
    
    def test_check_duplicates(self, sample_data):
        """Test duplicate detection"""
        checker = DataQualityChecker()
        
        # Add duplicate
        duplicate_data = pd.concat([sample_data, sample_data.iloc[:5]])
        
        result = checker.check_duplicates(duplicate_data, subset=['claim_id'])
        
        assert result > 0
        assert result == 5
    
    def test_check_duplicates_no_duplicates(self, sample_data):
        """Test when no duplicates exist"""
        checker = DataQualityChecker()
        
        result = checker.check_duplicates(sample_data, subset=['claim_id'])
        
        assert result == 0
    
    def test_check_outliers_iqr(self, sample_data):
        """Test IQR-based outlier detection"""
        checker = DataQualityChecker()
        
        # Add obvious outlier
        sample_data.loc[0, 'claim_amount'] = 1000000  # $1M claim
        
        outliers = checker.check_outliers(sample_data, 'claim_amount', method='iqr')
        
        assert len(outliers) > 0
        assert 0 in outliers.index
    
    def test_check_outliers_zscore(self, sample_data):
        """Test z-score based outlier detection"""
        checker = DataQualityChecker()
        
        # Add outlier
        sample_data.loc[0, 'claim_amount'] = 1000000
        
        outliers = checker.check_outliers(sample_data, 'claim_amount', method='zscore', threshold=3)
        
        assert len(outliers) > 0
    
    def test_check_data_types(self, sample_data):
        """Test data type validation"""
        checker = DataQualityChecker()
        
        expected_types = {
            'claim_id': 'object',
            'claim_amount': 'float64',
            'claim_type': 'object',
            'vehicle_age': 'int64'
        }
        
        result = checker.check_data_types(sample_data, expected_types)
        
        assert isinstance(result, dict)
        assert all(v == True for v in result.values())
    
    def test_check_data_types_mismatch(self, sample_data):
        """Test data type mismatch detection"""
        checker = DataQualityChecker()
        
        expected_types = {
            'claim_amount': 'int64',  # Wrong type
        }
        
        result = checker.check_data_types(sample_data, expected_types)
        
        assert result['claim_amount'] == False
    
    def test_check_value_ranges(self, sample_data):
        """Test value range validation"""
        checker = DataQualityChecker()
        
        ranges = {
            'claim_amount': (0, 100000),
            'vehicle_age': (0, 30),
            'driver_age': (16, 100)
        }
        
        violations = checker.check_value_ranges(sample_data, ranges)
        
        assert isinstance(violations, dict)
    
    def test_check_categorical_values(self, sample_data):
        """Test categorical value validation"""
        checker = DataQualityChecker()
        
        valid_values = {
            'claim_type': ['Collision', 'Comprehensive', 'Liability']
        }
        
        result = checker.check_categorical_values(sample_data, valid_values)
        
        assert isinstance(result, dict)
        assert 'claim_type' in result
        assert result['claim_type']['invalid_count'] == 0
    
    def test_check_categorical_values_invalid(self):
        """Test detection of invalid categorical values"""
        checker = DataQualityChecker()
        
        df = pd.DataFrame({
            'claim_type': ['Collision', 'Invalid', 'Comprehensive', 'BadValue']
        })
        
        valid_values = {
            'claim_type': ['Collision', 'Comprehensive', 'Liability']
        }
        
        result = checker.check_categorical_values(df, valid_values)
        
        assert result['claim_type']['invalid_count'] == 2
    
    def test_calculate_quality_score_perfect(self, sample_data):
        """Test quality score for perfect data"""
        checker = DataQualityChecker()
        
        score = checker.calculate_quality_score(sample_data)
        
        assert 0 <= score <= 100
        assert score >= 90  # Should be high for clean data
    
    def test_calculate_quality_score_poor(self, data_with_nulls):
        """Test quality score for poor quality data"""
        checker = DataQualityChecker()
        
        score = checker.calculate_quality_score(data_with_nulls)
        
        assert 0 <= score <= 100
        assert score < 90  # Should be lower due to missing values


class TestDataQualityIntegration:
    """Integration tests for data quality workflow"""
    
    def test_full_quality_check_workflow(self):
        """Test complete data quality check pipeline"""
        np.random.seed(42)
        
        # Create realistic claims data
        df = pd.DataFrame({
            'claim_id': [f'CLM-{i:05d}' for i in range(1000)],
            'claim_amount': np.random.lognormal(9, 1, 1000),
            'claim_type': np.random.choice(['Collision', 'Comprehensive'], 1000),
            'vehicle_age': np.random.randint(0, 20, 1000),
            'driver_age': np.random.randint(18, 80, 1000),
            'fraud_flag': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        })
        
        # Add some quality issues
        df.loc[0:10, 'claim_amount'] = np.nan  # Missing values
        df.loc[900:905, :] = df.loc[900, :]  # Duplicates
        df.loc[20, 'claim_amount'] = 1000000  # Outlier
        
        checker = DataQualityChecker()
        
        # Run all checks
        missing = checker.check_missing_values(df)
        duplicates = checker.check_duplicates(df)
        outliers = checker.check_outliers(df, 'claim_amount')
        score = checker.calculate_quality_score(df)
        
        # Verify results
        assert missing['claim_amount'] > 0
        assert duplicates > 0
        assert len(outliers) > 0
        assert 0 <= score <= 100
    
    def test_quality_report_generation(self):
        """Test comprehensive quality report"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'claim_amount': np.random.uniform(1000, 50000, 100),
            'claim_type': np.random.choice(['Collision', 'Comprehensive'], 100),
            'fraud_flag': np.random.choice([0, 1], 100)
        })
        
        checker = DataQualityChecker()
        
        report = {
            'missing_values': checker.check_missing_values(df),
            'duplicates': checker.check_duplicates(df),
            'quality_score': checker.calculate_quality_score(df)
        }
        
        assert 'missing_values' in report
        assert 'duplicates' in report
        assert 'quality_score' in report
        assert isinstance(report['quality_score'], (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
