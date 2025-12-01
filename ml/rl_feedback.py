"""
RL Feedback Learning System for DQN agent.
Allows the RL agent to learn from actual trading outcomes and improve over time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ml.trainers import TradingEnv, DQN_CONFIG, DQN_TOTAL_TIMESTEPS


class RLFeedbackLearner:
    """
    Manages feedback learning for DQN agent.
    Collects feedback from actual predictions/trades and retrains the agent.
    """
    
    def __init__(self, feedback_dir: Optional[Path] = None):
        """
        Initialize RL feedback learner.
        
        Args:
            feedback_dir: Directory to store feedback data and retrained models
        """
        self.feedback_dir = feedback_dir or Path("data/rl_feedback")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.feedback_dir / "feedback_history.jsonl"
        self.retrain_history_file = self.feedback_dir / "retrain_history.json"
        
        # Load retrain history
        self.retrain_history = self._load_retrain_history()
    
    def _load_retrain_history(self) -> Dict[str, Any]:
        """Load retrain history."""
        if self.retrain_history_file.exists():
            try:
                with open(self.retrain_history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {
            "last_retrain": None,
            "retrain_count": 0,
            "feedback_count": 0
        }
    
    def _save_retrain_history(self):
        """Save retrain history."""
        try:
            with open(self.retrain_history_file, "w", encoding="utf-8") as f:
                json.dump(self.retrain_history, f, indent=2)
        except Exception as exc:
            print(f"[RL_FEEDBACK] Failed to save retrain history: {exc}")
    
    def add_feedback(
        self,
        symbol: str,
        asset_type: str,
        timeframe: str,
        prediction_timestamp: str,
        action_taken: str,  # long, short, hold
        predicted_return: float,
        predicted_price: float,
        actual_price: Optional[float] = None,
        actual_return: Optional[float] = None,
        reward: Optional[float] = None,
        features: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add feedback for a prediction/trade.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            timeframe: Timeframe
            prediction_timestamp: When prediction was made
            action_taken: Action that was taken (long/short/hold)
            predicted_return: Predicted return
            predicted_price: Predicted price
            actual_price: Actual price at horizon (optional)
            actual_return: Actual return at horizon (optional)
            reward: Calculated reward (optional, will be computed if not provided)
            features: Feature vector used for prediction (optional)
            metadata: Additional metadata
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "prediction_timestamp": prediction_timestamp,
            "action_taken": action_taken,
            "predicted_return": float(predicted_return),
            "predicted_price": float(predicted_price),
            "actual_price": float(actual_price) if actual_price is not None else None,
            "actual_return": float(actual_return) if actual_return is not None else None,
            "reward": float(reward) if reward is not None else None,
            "features": features or {},
            "metadata": metadata or {}
        }
        
        # Calculate reward if not provided and we have actual return
        if entry["reward"] is None and entry["actual_return"] is not None:
            entry["reward"] = self._calculate_reward(
                action_taken, entry["predicted_return"], entry["actual_return"]
            )
        
        # Append to feedback file
        try:
            with open(self.feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            self.retrain_history["feedback_count"] += 1
            self._save_retrain_history()
        except Exception as exc:
            print(f"[RL_FEEDBACK] Failed to save feedback: {exc}")
    
    def _calculate_reward(
        self,
        action: str,
        predicted_return: float,
        actual_return: float,
        transaction_cost: float = 0.0005
    ) -> float:
        """
        Calculate reward based on action and actual outcome.
        
        Args:
            action: Action taken (long/short/hold)
            predicted_return: Predicted return
            actual_return: Actual return
            transaction_cost: Transaction cost
        
        Returns:
            Reward value
        """
        if action == "hold":
            return 0.0
        
        # Map action to position
        position = 1.0 if action == "long" else -1.0 if action == "short" else 0.0
        
        # Calculate PnL
        raw_pnl = position * actual_return
        cost = transaction_cost if action != "hold" else 0.0
        net_pnl = raw_pnl - cost
        
        # Use tanh to normalize reward
        reward = float(np.tanh(net_pnl / 0.01))
        
        return reward
    
    def get_feedback_data(
        self,
        symbol: Optional[str] = None,
        asset_type: Optional[str] = None,
        min_feedback_count: int = 50
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Get feedback data for retraining.
        
        Args:
            symbol: Filter by symbol (optional)
            asset_type: Filter by asset type (optional)
            min_feedback_count: Minimum number of feedback entries needed
        
        Returns:
            Tuple of (feedback_entries, has_enough_data)
        """
        if not self.feedback_file.exists():
            return [], False
        
        entries = []
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        
                        # Filter
                        if symbol and entry.get("symbol") != symbol:
                            continue
                        if asset_type and entry.get("asset_type") != asset_type:
                            continue
                        
                        # Only include entries with actual return (completed feedback)
                        if entry.get("actual_return") is not None:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        has_enough = len(entries) >= min_feedback_count
        return entries, has_enough
    
    def retrain_dqn_with_feedback(
        self,
        symbol: str,
        asset_type: str,
        timeframe: str,
        base_model_path: Path,
        feature_columns: List[str],
        retrain_steps: int = 1000
    ) -> Optional[Path]:
        """
        Retrain DQN model with feedback data.
        
        Args:
            symbol: Trading symbol
            asset_type: Asset type
            timeframe: Timeframe
            base_model_path: Path to base trained model
            feature_columns: List of feature column names
            retrain_steps: Number of training steps for retraining
        
        Returns:
            Path to retrained model, or None if retraining failed
        """
        # Get feedback data
        feedback_entries, has_enough = self.get_feedback_data(
            symbol=symbol,
            asset_type=asset_type,
            min_feedback_count=20  # Lower threshold for retraining
        )
        
        if not has_enough:
            print(f"[RL_FEEDBACK] Not enough feedback data for {symbol} (need at least 20, got {len(feedback_entries)})")
            return None
        
        try:
            # Load base model
            if not base_model_path.exists():
                print(f"[RL_FEEDBACK] Base model not found: {base_model_path}")
                return None
            
            base_model = DQN.load(str(base_model_path))
            
            # Prepare training data from feedback
            observations = []
            actions = []
            rewards = []
            
            for entry in feedback_entries:
                features = entry.get("features", {})
                if not features:
                    continue
                
                # Build feature vector in correct order
                feature_vector = [features.get(col, 0.0) for col in feature_columns]
                observations.append(feature_vector)
                
                # Map action to action ID
                action_map = {"short": 0, "hold": 1, "long": 2}
                action_taken = entry.get("action_taken", "hold")
                actions.append(action_map.get(action_taken, 1))
                
                # Get reward
                reward = entry.get("reward", 0.0)
                if reward is None:
                    reward = 0.0
                rewards.append(reward)
            
            if len(observations) < 10:
                print(f"[RL_FEEDBACK] Not enough valid feedback entries for retraining")
                return None
            
            # Convert to numpy arrays
            obs_array = np.array(observations, dtype=np.float32)
            action_array = np.array(actions, dtype=np.int32)
            reward_array = np.array(rewards, dtype=np.float32)
            
            # Create environment from feedback data
            # TradingEnv expects returns, but we'll use rewards as returns for simplicity
            # In practice, we'd reconstruct returns from actual_price/predicted_price
            returns_array = reward_array  # Use rewards as proxy for returns
            
            def _make_feedback_env():
                return Monitor(TradingEnv(obs_array, returns_array))
            
            vec_env = DummyVecEnv([_make_feedback_env])
            
            # Create new model with same architecture
            retrain_model = DQN(
                "MlpPolicy",
                vec_env,
                verbose=0,
                tensorboard_log="logs/tensorboard",
                **DQN_CONFIG,
            )
            
            # Transfer weights from base model
            retrain_model.policy.load_state_dict(base_model.policy.state_dict())
            retrain_model.target_q_net.load_state_dict(base_model.target_q_net.state_dict())
            
            # Retrain with feedback data
            print(f"[RL_FEEDBACK] Retraining DQN with {len(observations)} feedback entries...")
            retrain_model.learn(total_timesteps=retrain_steps, progress_bar=False)
            
            # Save retrained model
            retrained_model_path = (
                self.feedback_dir / f"{asset_type}_{symbol}_{timeframe}_retrained_dqn.zip"
            )
            retrain_model.save(str(retrained_model_path))
            
            # Update history
            self.retrain_history["last_retrain"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self.retrain_history["retrain_count"] += 1
            self._save_retrain_history()
            
            print(f"[RL_FEEDBACK] Retrained model saved to {retrained_model_path}")
            return retrained_model_path
        
        except Exception as exc:
            print(f"[RL_FEEDBACK] Failed to retrain DQN: {exc}")
            return None


# Global feedback learner instance
_feedback_learner: Optional[RLFeedbackLearner] = None


def get_feedback_learner() -> RLFeedbackLearner:
    """Get or create global feedback learner instance."""
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = RLFeedbackLearner()
    return _feedback_learner

