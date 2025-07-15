ENHANCED_ACTION_SPACE = [
    (-1, 1, 0.2),   # Left + Gas + Light Brake
    (0, 1, 0.2),    # Straight + Gas + Light Brake  
    (1, 1, 0.2),    # Right + Gas + Light Brake
    (-1, 1, 0),     # Left + Gas
    (0, 1, 0),      # Straight + Gas
    (1, 1, 0),      # Right + Gas
    (-1, 0, 0.2),   # Left + Light Brake
    (0, 0, 0.2),    # Straight + Light Brake
    (1, 0, 0.2),    # Right + Light Brake
    (-1, 0, 0),     # Left only
    (0, 0, 0),      # Do nothing
    (1, 0, 0)       # Right only
]

def get_action_from_index(action_index):
    """Convert action index to (steering, gas, brake) tuple"""
    return ENHANCED_ACTION_SPACE[action_index]

def get_num_actions():
    """Return number of discrete actions"""
    return len(ENHANCED_ACTION_SPACE)