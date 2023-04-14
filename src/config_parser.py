from pathlib import Path
import toml

def load_config(config_name: str) -> dict:
    """
    Loads and returns the configuration file specified by passed argument.
    Args:
        config_name (str): Name of the configuration file present in the configs directory.

    Returns:
        dict: The required configuration file.
    """
    # Navigate to the config directory
    config_dir = Path(__file__).parent.parent/ "configs"
    # Construct the path to the configuration file
    config_file = config_dir / f"{config_name}.toml"
    # Use the `open()` method to open the file, then use the `toml.load()` function
    # to parse its contents
    with config_file.open() as f:
        config = toml.load(f)
    return config