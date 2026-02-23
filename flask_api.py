"""
Insurance Claims Analysis - REST API
=====================================
Production-ready Flask API for claim severity prediction and fraud detection

Deploy with: gunicorn -w 4 -b 0.0.0.0:5000 api:app
"""

from flask import Flask, request, jsonify
import pandas as pd
import logging
from datetime import datetime
from insurance_claims_analysis import (
    ClaimsAnalysisEngine,
    ClaimAnalysisConfig,
    ClaimType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model (in production, load from saved model)
config = ClaimAnalysisConfig(
    claim_type=ClaimType.AUTO,
    target_variable='claim_amount',
    categorical_features=['claim_type', 'vehicle_age', 'driver_age_group', 'region'],
    numerical_features=['policy_age', 'num_claims_history', 'time_to_report', 'vehicle_value'],
    fraud_indicators=['time_to_report', 'policy_age', 'num_claims_history'],
    use_xgboost=False
)

# Initialize engine (would load trained model in production)
engine = ClaimsAnalysisEngine(config)

# Metrics for monitoring
prediction_count = 0
fraud_flag_count = 0


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'predictions_served': prediction_count
    })


@app.route('/predict', methods=['POST'])
def predict_claim():
    """
    Predict claim severity and detect fraud
    
    Request body:
    {
        "claim_id": "CLM-001234",
        "claim_type": "Collision",
        "vehicle_age": "4-7 years",
        "driver_age_group": "36-50",
        "region": "Urban",
        "policy_age": 2.5,
        "num_claims_history": 1,
        "time_to_report": 1.2,
        "vehicle_value": 25000
    }
    
    Response:
    {
        "claim_id": "CLM-001234",
        "predicted_severity": 12450.75,
        "confidence_interval": {
            "lower": 10200.50,
            "upper": 14701.00
        },
        "fraud_analysis": {
            "risk_level": "low",
            "probability": 0.23,
            "suspicious_features": []
        },
        "recommendation": "Standard processing",
        "timestamp": "2024-02-23T10:30:00"
    }
    """
    global prediction_count, fraud_flag_count
    
    try:
        # Parse request
        data = request.json
        claim_id = data.get('claim_id', 'unknown')
        
        logger.info(f"Processing prediction request for claim {claim_id}")
        
        # Validate required fields
        required_fields = [
            'claim_type', 'vehicle_age', 'driver_age_group', 'region',
            'policy_age', 'num_claims_history', 'time_to_report', 'vehicle_value'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Create DataFrame
        claim_df = pd.DataFrame([data])
        
        # Make prediction
        predictions = engine.predict_claim(claim_df)
        pred = predictions[0]
        
        # Track metrics
        prediction_count += 1
        if pred.fraud_analysis.fraud_risk_level.value in ['high', 'critical']:
            fraud_flag_count += 1
        
        # Format response
        response = {
            'claim_id': claim_id,
            'predicted_severity': round(pred.predicted_severity, 2),
            'confidence_interval': {
                'lower': round(pred.confidence_interval[0], 2),
                'upper': round(pred.confidence_interval[1], 2)
            },
            'fraud_analysis': {
                'risk_level': pred.fraud_analysis.fraud_risk_level.value,
                'probability': round(pred.fraud_analysis.fraud_probability, 3),
                'suspicious_features': [
                    {'feature': f, 'score': round(s, 2)}
                    for f, s in pred.fraud_analysis.suspicious_features
                ]
            },
            'top_cost_drivers': [
                {'feature': f, 'importance': round(imp, 4)}
                for f, imp in pred.cost_drivers
            ],
            'recommendation': pred.recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction complete for {claim_id}: ${pred.predicted_severity:.2f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Request body:
    {
        "claims": [
            {...claim1...},
            {...claim2...},
            ...
        ]
    }
    """
    try:
        data = request.json
        claims = data.get('claims', [])
        
        if not claims:
            return jsonify({'error': 'No claims provided'}), 400
        
        logger.info(f"Processing batch prediction for {len(claims)} claims")
        
        # Create DataFrame
        claims_df = pd.DataFrame(claims)
        
        # Make predictions
        predictions = engine.predict_claim(claims_df)
        
        # Format response
        results = []
        for i, pred in enumerate(predictions):
            claim_id = claims[i].get('claim_id', f'claim_{i}')
            
            results.append({
                'claim_id': claim_id,
                'predicted_severity': round(pred.predicted_severity, 2),
                'fraud_risk': pred.fraud_analysis.fraud_risk_level.value,
                'recommendation': pred.recommendation
            })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Metrics endpoint for monitoring"""
    return jsonify({
        'total_predictions': prediction_count,
        'fraud_flags': fraud_flag_count,
        'fraud_flag_rate': fraud_flag_count / prediction_count if prediction_count > 0 else 0,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=False)


"""
DEPLOYMENT EXAMPLES
===================

1. Development:
   python api.py

2. Production (Gunicorn):
   gunicorn -w 4 -b 0.0.0.0:5000 api:app

3. Docker:
   docker build -t claims-api .
   docker run -p 5000:5000 claims-api

4. Kubernetes:
   kubectl apply -f deployment.yaml

EXAMPLE REQUESTS
================

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-001",
    "claim_type": "Collision",
    "vehicle_age": "4-7 years",
    "driver_age_group": "36-50",
    "region": "Urban",
    "policy_age": 2.5,
    "num_claims_history": 1,
    "time_to_report": 1.2,
    "vehicle_value": 25000
  }'

# Batch prediction
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      {"claim_id": "CLM-001", ...},
      {"claim_id": "CLM-002", ...}
    ]
  }'

# Health check
curl http://localhost:5000/health

# Metrics
curl http://localhost:5000/metrics

MONITORING
==========

Integrate with:
- Prometheus for metrics
- CloudWatch for AWS
- Datadog for full observability
- ELK stack for log aggregation
"""
