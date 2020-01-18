"""
Exceptions for SQL to Pandas
"""


class MultipleQueriesException(Exception):
    """
    Raised when multiple queries are passed into sql to pandas.
    """

    def __init__(self):
        super(MultipleQueriesException, self).__init__("Only one sql statement may be entered")


class InvalidQueryException(Exception):
    """
    Raised when an invalid query is passed into a sql to pandas.
    """
    def __init__(self, message):
        super(InvalidQueryException, self).__init__(f"Invalid query!\n" + message)


class DataFrameDoesNotExist(Exception):
    """
    Raised when a DataFrame doesn't exist
    """

    def __init__(self, table_name):
        super(DataFrameDoesNotExist, self).__init__(f"DataFrame {table_name} has not been defined")
