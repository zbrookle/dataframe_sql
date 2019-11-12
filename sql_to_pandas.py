"""
Convert sql statement to run on pandas dataframes
"""
from sqlparse import split
from lark import Lark
from lark.exceptions import UnexpectedCharacters

with open(file="sql.grammar") as sql_grammar_file:
    GRAMMAR_TEXT = sql_grammar_file.read()
    GRAMMAR = Lark(GRAMMAR_TEXT)


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
        super(InvalidQueryException, self).__init__(message)


class SqlToPandas:
    """
    Class that handles conversion from sql to pandas data frame methods.
    """
    def __init__(self, sql: str, all_global_vars):
        self.sql = sql
        self.verify_sql()
        self.ast = self.parse_sql()
        print(self.ast)

    def verify_sql(self):
        """
        Verifies that the sql is ONE valid statement
        :param sql: A sql query
        :return:
        """
        split_sql = split(self.sql)
        if len(split_sql) > 1:
            raise MultipleQueriesException

    def parse_sql(self):
        """
        Splits the sql into tokens
        :return:
        """
        # TODO Put more specific errors
        try:
            return GRAMMAR.parse(self.sql)
        except UnexpectedCharacters as err:
            message_begin = "Invalid query!"
            message_reason = ""
            message_loc = f"Unexpected input at line {err.line}, column {err.column}."
            if err.allowed == {'SELECT'}:
                message_reason = "Not a select statement."
            raise InvalidQueryException(f"{message_begin} {message_reason} {message_loc} \nQuery:"
                                        f"\n{err.get_context(self.sql)}")

    def execute_sql(self):
        """
        Run the query that was pass into the sql variable
        :return:
        """
        print(self.ast)
