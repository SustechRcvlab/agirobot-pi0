"""
Custom transform for replacing gripper data in state with normalized action data.
"""

import dataclasses
import numpy as np
from openpi.transforms import DataTransformFn


@dataclasses.dataclass(frozen=True)
class ReplaceGripperInState(DataTransformFn):
    """Replace gripper dimensions in state with corresponding action values.
    
    This transform replaces specific dimensions in the state vector with values
    from the actions vector. This is useful when you want to use action gripper
    commands as the current gripper state.
    
    Args:
        gripper_dims: List of dimensions in state/actions that correspond to grippers.
                     For example, [7, 15] means dimensions 7 and 15 are gripper values.
        use_action_values: If True, use action values to replace state values.
                          If False, this transform is a no-op.
    """
    
    gripper_dims: list[int] = dataclasses.field(default_factory=lambda: [7, 15])
    use_action_values: bool = True
    
    def __call__(self, data: dict) -> dict:
        if not self.use_action_values or "state" not in data or "actions" not in data:
            return data
        
        # Make a copy to avoid modifying the original data
        data = data.copy()
        state = np.copy(data["state"])
        actions = data["actions"]
        
        # Replace gripper dimensions in state with corresponding action values
        # We use the first action timestep for the current state
        if actions.ndim > 1:
            # If actions is a sequence, use the first timestep
            action_values = actions[0]
        else:
            # If actions is a single timestep
            action_values = actions
            
        for dim in self.gripper_dims:
            if dim < len(state) and dim < len(action_values):
                state[dim] = action_values[dim]
                #print(f"Replaced state[{dim}] with action[{dim}]: {state[dim]}")
        
        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True) 
class ReplaceGripperInStatePostNorm(DataTransformFn):
    """Replace gripper dimensions in state with normalized action values after normalization.
    
    This transform should be applied AFTER normalization to ensure both state and action
    gripper values are normalized before replacement.
    
    Args:
        gripper_dims: List of dimensions that correspond to grippers.
        use_action_values: If True, use action values to replace state values.
    """
    
    gripper_dims: list[int] = dataclasses.field(default_factory=lambda: [7, 15])
    use_action_values: bool = True
    
    def __call__(self, data: dict) -> dict:
        if not self.use_action_values or "state" not in data or "actions" not in data:
            return data
        
        # Make a copy to avoid modifying the original data  
        data = data.copy()
        state = np.copy(data["state"])
        actions = data["actions"]
        
        # Replace gripper dimensions in normalized state with normalized action values
        if actions.ndim > 1:
            # If actions is a sequence, use the first timestep
            action_values = actions[0]
        else:
            # If actions is a single timestep
            action_values = actions
            
        for dim in self.gripper_dims:
            if dim < len(state) and dim < len(action_values):
                old_value = state[dim]
                state[dim] = action_values[dim]
                #print(f"Post-norm: Replaced state[{dim}] ({old_value:.4f}) with action[{dim}] ({action_values[dim]:.4f})")
        
        data["state"] = state
        return data
