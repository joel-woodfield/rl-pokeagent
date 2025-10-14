#------------------------------------------------------------------------------
# STATE FORMATTERS
#------------------------------------------------------------------------------

def format_state_for_llm(game_state):
    return "TEST"


#------------------------------------------------------------------------------
# HELPERS
#------------------------------------------------------------------------------

def save_persistent_world_map(file_path=None):
    """Deprecated - MapStitcher handles all persistence now"""
    # MapStitcher auto-saves, nothing to do here
    pass

def load_persistent_world_map(file_path=None):
    """Deprecated - MapStitcher handles all persistence now"""
    # MapStitcher auto-loads, nothing to do here
    pass