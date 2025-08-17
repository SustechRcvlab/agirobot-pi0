"""
Test script to verify gripper replacement functionality.
"""

import numpy as np
import sys
import os
sys.path.append('/app/src')

from openpi.transforms_gripper_replacement import ReplaceGripperInState, ReplaceGripperInStatePostNorm

def test_gripper_replacement():
    """Test the gripper replacement transforms."""
    
    # Create sample data
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.1, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 0.2])
    actions = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 0.9, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 0.8])
    
    data = {
        "state": state,
        "actions": actions
    }
    
    print("Original data:")
    print(f"State: {state}")
    print(f"Actions: {actions}")
    print(f"State gripper dims [7, 15]: [{state[7]}, {state[15]}]")
    print(f"Action gripper dims [7, 15]: [{actions[7]}, {actions[15]}]")
    
    # Test pre-normalization replacement
    transform_pre = ReplaceGripperInState(gripper_dims=[7, 15], use_action_values=True)
    data_transformed = transform_pre(data.copy())
    
    print("\nAfter pre-normalization replacement:")
    print(f"State: {data_transformed['state']}")
    print(f"State gripper dims [7, 15]: [{data_transformed['state'][7]}, {data_transformed['state'][15]}]")
    
    # Test post-normalization replacement
    transform_post = ReplaceGripperInStatePostNorm(gripper_dims=[7, 15], use_action_values=True)
    data_transformed_post = transform_post(data.copy())
    
    print("\nAfter post-normalization replacement:")
    print(f"State: {data_transformed_post['state']}")
    print(f"State gripper dims [7, 15]: [{data_transformed_post['state'][7]}, {data_transformed_post['state'][15]}]")
    
    # Test with sequence actions
    actions_seq = np.array([[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 0.9, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 0.8],
                           [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 0.95, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 0.85]])
    
    data_seq = {
        "state": state.copy(),
        "actions": actions_seq
    }
    
    print("\nWith action sequence (should use first timestep):")
    print(f"Actions shape: {actions_seq.shape}")
    print(f"First action gripper dims [7, 15]: [{actions_seq[0][7]}, {actions_seq[0][15]}]")
    
    data_seq_transformed = transform_post(data_seq)
    print(f"State gripper dims after replacement [7, 15]: [{data_seq_transformed['state'][7]}, {data_seq_transformed['state'][15]}]")

if __name__ == "__main__":
    test_gripper_replacement()
