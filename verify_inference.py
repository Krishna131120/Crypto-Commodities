
import sys
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from ml.inference import InferencePipeline, RiskManagerConfig

def run_verification():
    print("=" * 80)
    print("VERIFYING INFERENCE PIPELINE")
    print("=" * 80)

    # 1. Setup Mock Models
    print("\n[1] SETTING UP MOCK MODELS...")
    
    # Mock model directory and files
    model_dir = Path("mock_models/crypto/BTC-USDT/1d/short")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy scaler mock
    scaler = MagicMock()
    scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
    scaler.feature_names_in_ = ["f1", "f2", "f3"]
    
    # Create dummy models
    model_rf = MagicMock()
    model_rf.predict.return_value = np.array([0.02]) # Predicts 2% return
    
    model_xgb = MagicMock()
    model_xgb.predict.return_value = np.array([0.025]) # Predicts 2.5% return
    
    # Mock joblib.load to return our mocks
    with patch("joblib.load") as mock_load:
        def side_effect(path):
            if "feature_scaler" in str(path):
                return scaler
            elif "rf" in str(path):
                return model_rf
            elif "xgb" in str(path):
                return model_xgb
            return MagicMock()
            
        mock_load.side_effect = side_effect
        
        # Mock glob to find our "files"
        with patch.object(Path, "glob", side_effect=[
            [Path("feature_scaler.joblib"), Path("rf.joblib"), Path("xgb.joblib")] 
        ]):
             # Initialize pipeline
            pipeline = InferencePipeline(
                model_dir=model_dir,
                risk_config=RiskManagerConfig(),
                asset_type="crypto" # Explicitly set for test
            )
            
            # Manually inject models since glob patch might be tricky with pathlib
            pipeline.scaler = scaler
            pipeline.models = {"rf": model_rf, "xgb": model_xgb}
            # Add synthetic metrics
            pipeline.metrics = {
                "rf": {"r2": 0.4, "directional_accuracy": 0.6},
                "xgb": {"r2": 0.5, "directional_accuracy": 0.65}
            }
            pipeline.loaded = True
            
            # 2. Test Prediction Verification
            print("\n[2] TESTING ENSEMBLE PREDICTION...")
            feature_row = pd.Series({"f1": 1, "f2": 2, "f3": 3})
            current_price = 100.0
            volatility = 0.02
            
            result = pipeline.predict(feature_row, current_price, volatility)
            consensus = result["consensus"]
            
            print(f"  Consensus Action: {consensus['consensus_action']}")
            print(f"  Consensus Return: {consensus['consensus_return']:.4f}")
            print(f"  Confidence:       {consensus['consensus_confidence']:.2f}")
            
            # Validation logic
            # Avg return should be (0.02 + 0.025) / 2 = 0.0225 
            # Note: The pipeline clamps predictions and might apply weighting
            if 0.020 <= consensus["consensus_return"] <= 0.023:
                 print("  [OK] Consensus return calculation verified")
            else:
                 print(f"  [FAIL] Unexpected consensus return: {consensus['consensus_return']}")

            # 3. Test DQN Integration (if simple models work, assume structure holds, but let's test integration logic)
            print("\n[3] TESTING DQN INTEGRATION...")
            # Inject DQN result via summary
            pipeline.summary = {
                "model_predictions": {
                    "dqn": {
                        "predicted_return_pct": 5.0, # Strong long
                        "action": "long",
                        "confidence": 80.0
                    }
                }
            }
            
            result_with_dqn = pipeline.predict(feature_row, current_price, volatility)
            dqn_cons = result_with_dqn["consensus"]
            print(f"  Consensus with DQN: {dqn_cons['consensus_return']:.4f}")
            
            # DQN should pull the average up. 
            # Avg of 0.02, 0.025, and 0.05 is ~0.031
            if dqn_cons["consensus_return"] > consensus["consensus_return"]:
                print("  [OK] DQN influenced consensus positively")
            else:
                print("  [FAIL] DQN did not influence consensus")

            # 4. Test Intraday Price Action Override
            print("\n[4] TESTING INTRADAY OVERRIDE...")
            # Set profile to intraday to enable the logic
            pipeline.horizon_profile = "intraday"
            
            # Scenario: Models predict small gain (2%), but price jumped 5% (Strong Momentum)
            # Feature row with previous close
            prev_close = 100.0
            curr_price = 105.0 # +5% move
            feature_row_intraday = pd.Series({"f1": 1, "f2": 2, "f3": 3, "Close_Lag_1": prev_close})
            
            # Repatch models to predict small gain (0.01)
            model_rf.predict.return_value = np.array([0.01])
            model_xgb.predict.return_value = np.array([0.01])
            pipeline.models = {"rf": model_rf, "xgb": model_xgb}
            pipeline.summary = {} # Clear DQN
            
            result_override = pipeline.predict(feature_row_intraday, curr_price, volatility)
            cons_override = result_override["consensus"]
            
            print(f"  Model Prediction: ~0.01")
            print(f"  Price Move:       +0.05 (5%)")
            print(f"  Final Consensus:  {cons_override['consensus_return']:.4f}")
            
            # The override logic fuses model pred (0.01) with price move (0.05)
            # Should be significantly higher than 0.01
            if cons_override["consensus_return"] > 0.02:
                 print("  [OK] Intraday price action overrode model prediction")
            else:
                 print("  [FAIL] Price action did not override model prediction")

    # Clean up mock dir
    import shutil
    if Path("mock_models").exists():
        shutil.rmtree("mock_models")
    
    print("\nVERIFICATION COMPLETE")

if __name__ == "__main__":
    run_verification()
