"""
Convert sql statement to run on pandas dataframes
"""
import os
from pathlib import Path

from lark import Lark, UnexpectedToken
from sqlparse import split

from dataframe_sql.exceptions.sql_exception import (
    InvalidQueryException,
    MultipleQueriesException,
)
from dataframe_sql.parsers import SQLTransformer

SHOW_TREE = False
SHOW_DF = False
_ROOT = Path(__file__).parent
GRAMMAR_PATH = os.path.join(_ROOT, "grammar", "sql.grammar")
with open(file=GRAMMAR_PATH) as sql_grammar_file:
    GRAMMAR_TEXT = sql_grammar_file.read()


class SqlToPandas:
    """
    Class that handles conversion from sql to pandas data frame methods.
    """
    def __init__(self, sql: str, all_global_vars):
        self.sql = sql
        self.verify_sql()
        if SHOW_TREE:
            self.parser = Lark(GRAMMAR_TEXT, parser='lalr')
        else:
            self.parser = Lark(GRAMMAR_TEXT, parser='lalr', transformer=SQLTransformer(all_global_vars))
        self.ast = self.parse_sql()
        if SHOW_TREE or SHOW_DF:
            print("Result:")
        if SHOW_TREE:
            print(self.ast)
            print(self.ast.pretty())
        if SHOW_DF:
            print(self.ast)
        self.data_frame = self.ast

    def verify_sql(self):
        """
        Verifies that the sql is ONE valid statement
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
        try:
            return self.parser.parse(self.sql)
        except UnexpectedToken as err:
            message = f"Expected one of the following input(s): {err.expected}\n" \
                      f"Unexpected input at line {err.line}, column {err.column}\n" \
                      f"{err.get_context(self.sql)}"
            raise InvalidQueryException(message)
