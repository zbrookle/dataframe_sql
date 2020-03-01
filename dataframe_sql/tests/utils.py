"""
Shared functions among the tests like setting up test environment
"""
from pathlib import Path

from pandas import DataFrame, read_csv

from dataframe_sql import register_temp_table, remove_temp_table

DATA_PATH = Path(__file__).parent.parent / "data"


# Import the data for testing
FOREST_FIRES = read_csv(DATA_PATH / "forestfires.csv")
DIGIMON_MON_LIST = read_csv(DATA_PATH / "DigiDB_digimonlist.csv")
DIGIMON_MOVE_LIST = read_csv(DATA_PATH / "DigiDB_movelist.csv")
DIGIMON_SUPPORT_LIST = read_csv(DATA_PATH / "DigiDB_supportlist.csv")
AVOCADO = read_csv(DATA_PATH / "avocado.csv")

# Name change is for name interference
DIGIMON_MON_LIST["mon_attribute"] = DIGIMON_MON_LIST["Attribute"]
DIGIMON_MOVE_LIST["move_attribute"] = DIGIMON_MOVE_LIST["Attribute"]


def register_env_tables():
    """
    Returns all globals but in lower case
    :return:
    """
    for variable_name in globals():
        variable = globals()[variable_name]
        if isinstance(variable, DataFrame):
            register_temp_table(frame=variable, table_name=variable_name)


def remove_env_tables():
    """
    Remove all env tables
    :return:
    """
    for variable_name in globals():
        variable = globals()[variable_name]
        if isinstance(variable, DataFrame):
            remove_temp_table(table_name=variable_name)
