"""
Exceptions for SQL to Pandas
"""


class MultipleQueriesException(Exception):
    """
    Raised when multiple queries are passed into sql to pandas.
    """

    def __init__(self):
        Exception.__init__(self, "Only one sql statement may be entered")


class InvalidQueryException(Exception):
    """
    Raised when an invalid query is passed into a sql to pandas.
    """

    def __init__(self, message):
        Exception.__init__(self, f"Invalid query!\n{message}")


class DataFrameDoesNotExist(Exception):
    """
    Raised when a DataFrame doesn't exist
    """

    def __init__(self, table_name):
        Exception.__init__(self, f"DataFrame {table_name} has not been defined")
