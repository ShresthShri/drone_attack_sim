#!/usr/bin/env python3
"""
Enhanced Drone Navigation Training Script
Run this script to start training with optimal parameters
"""

import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_drone_code import train, DroneNavConfig, test_model

def main():
    print("=" * 60)
    print("Enhanced Drone Navigation Training")
    print("=" * 60)
    
    # Train the model
    model, save_dir = train()
    
    print(f"\nTraining completed! Model saved to: {save_dir}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_model(f"{save_dir}/final_model.zip", num_episodes=5)
    
    print("\nTraining and testing complete!")

if __name__ == "__main__":
    main()
