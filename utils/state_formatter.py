import textwrap
from utils.map_formatter import format_map_grid, generate_legend_items

#------------------------------------------------------------------------------
# STATE FORMATTERS
#------------------------------------------------------------------------------

def format_state_for_llm(game_state):
    
    # Determine which state formatting to use
    game_data = game_state.get("game", {})
    is_in_battle = game_data.get("is_in_battle", False) or game_data.get("in_battle", False)
    if is_in_battle:
        formatted_state = "NOT CURRENTLY SUPPORTED"
    else:
        formatted_state = _format_state_for_llm_overworld(game_state)

    return formatted_state
    

def _format_state_for_llm_overworld(game_state):
    
    # Extract data
    player_data = game_state.get("player", {})
    game_data = game_state.get("game", {})
    map_data = game_state.get("map", {})

    name = player_data.get("name", "Unknown")
    money = game_data.get("money", "Unknown")

    region = "HOENN"
    location = player_data.get("location", "Unknown").replace("_", " ")
    position = player_data.get("position")
    x_coord = position.get("x", "?")
    y_coord = position.get("y", "?")

    raw_tiles = map_data.get('tiles', [])
    npcs = []
    player_coords = map_data.get('player_coords')
    map_grid = format_map_grid(raw_tiles, npcs, player_coords)
    map_rows = [" ".join(row) for row in map_grid]
    map_height = len(map_rows)
    map_width = len(map_grid[0])
    legend_items = generate_legend_items(map_grid)

    team = player_data.get("party", [])

    # Format state
    formatted_state = textwrap.dedent(f"""
    <game_state_data>
        <player_info>
            <name>{name}</name>
            <money>{money}</money>
            <in_battle>False</in_battle>
            <in_overworld>True</in_overworld>
        </player_info>
        <location>
            <region>{region}</region>
            <position x_coord={x_coord} y_coord={y_coord}>{location}</position>
            <traversability>
                <map height={map_height} width={map_width}>{"".join(list(map(lambda row: f'''
                    {row}''', map_rows)))}
                </map>
                <legend>{"".join(list(map(lambda item: f'''
                    {item}''', legend_items)))}
                </legend>
            <traversability>
        </location>
        <pokemon_team>
            {"Empty" if len(team) == 0 else "".join(list(map(lambda pokemon: f'''<pokemon>
                <species>{pokemon.get("species_name", "Unknown")}</species>
                <level>{pokemon.get("level", "Unknown")}</level>
                <hp>{pokemon.get("hp", "Unknown")}/{pokemon.get("max_hp", "Unknown")}</hp>
                <status>{pokemon.get("status", "OK")}</status>
            </pokemon>''', team)))}
        </pokemon_team>
    </game_state_data>    
    """)
    return formatted_state


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