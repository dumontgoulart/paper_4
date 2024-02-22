import os
import shutil

def copy_and_rename_gpkg(source_dir, target_dir, sim, climate_scenarios, adapt_scenarios):
    """
    Copy spatial.gpkg files from source directory to target directory,
    renaming them based on predefined scenario names.

    :param source_dir: The source directory containing scenario subdirectories.
    :param target_dir: The target directory where .gpkg files will be copied to.
    :param sim: The simulation prefix.
    :param climate_scenarios: List of climate scenarios.
    :param adapt_scenarios: List of adaptation scenarios.
    """
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Generate all combinations of climate and adaptation scenarios
    all_adapt_scenarios = [f"{sim}{climate}_rain_surge_{adapt}" for climate in climate_scenarios for adapt in adapt_scenarios]

    # Walk through the source directory to find matching scenario directories
    for scenario in all_adapt_scenarios:
        scenario_path = os.path.join(source_dir, scenario, 'Impacts', 'fiat_model', 'output', 'spatial.gpkg')
        if os.path.exists(scenario_path):
            new_filename = f"spatial_{scenario}.gpkg"
            target_path = os.path.join(target_dir, new_filename)
            
            # Copy the file to the new location with the new name
            shutil.copy2(scenario_path, target_path)
            print(f"Copied and renamed 'spatial.gpkg' to '{target_path}'")
        else:
            print(f"Scenario not found: {scenario_path}")

# Example usage
source_dir = r"D:\paper_4\data\FloodAdapt-GUI\Database\beira\output\Scenarios"
target_dir = r"D:\paper_4\data\floodadapt_results"
sim = 'idai_ifs_rebuild_bc_'
climate_scenarios = ['hist', '3c', 'hightide', '3c-hightide']
adapt_scenarios = ['noadapt', 'hold', 'retreat']

copy_and_rename_gpkg(source_dir, target_dir, sim, climate_scenarios, adapt_scenarios)
