import inspect
import re
import time
from types import FunctionType
from typing import List

from dataframe_sql.tests.pandas_sql_functionality_test import *  # noqa

DONT_TEST = [
    test_add_remove_temp_table,  # noqa
    test_for_valid_query,  # noqa
    test_for_non_existent_table,  # noqa
]
INDENT_REGEX = re.compile(r"(\t|\s{4})(?P<code>.*)")


def get_pandas_tests():
    global_dict = globals()
    test_list = []
    for global_key in global_dict:
        global_var = global_dict[global_key]
        if (
            isinstance(global_var, FunctionType)
            and "test" in global_var.__name__
            and global_var not in DONT_TEST
        ):
            test_list.append(global_var)

    return test_list


def get_function_code(function_object: FunctionType) -> List[str]:
    """
    Return the function code only for the function (not def name or docstring)
    :param function_object:
    :return:
    """
    doc_string = function_object.__doc__
    function_code = inspect.getsource(function_object)
    remove_doc_string = function_code.replace(f'"""{doc_string}"""', "")
    remove_function_name = remove_doc_string.split("\n")[1:]
    return remove_function_name


def fix_code_indent(function_code: List[str]):
    """
    Return code indented back one to account for the def indent
    :param function_code:
    :return:
    """
    for i, code_line in enumerate(function_code):
        match = INDENT_REGEX.match(code_line)
        if match:
            function_code[i] = match.group("code")


def remove_assertion(function_code: List[str]):
    """
    Remove assertion lines
    :param function_code:
    :return:
    """
    for i, code_line in enumerate(function_code):
        if code_line == "tm.assert_frame_equal(pandas_frame, my_frame)":
            function_code[i] = ""


def find_end_paren(function_code: str, start: int):
    """
    Find the end location given a starting parenthesis location
    :param function_code:
    :param start:
    :return:
    """
    parentheses = []
    for i, character in enumerate(function_code[start:]):
        if character == "(":
            parentheses.append(character)
        elif character == ")":
            parentheses.pop()
        if not parentheses:
            return i + start


def split_into_pandas_and_dataframe_sql(function_code: str):
    """
    Returns the code split into the half using dataframe_sql and the half using
    the direct pandas api
    :param function_code:
    :return:
    """
    data_frame_sql_code_init = "my_frame = query"
    dataframe_sql_code_start = function_code.find(data_frame_sql_code_init)
    text_offset = len(data_frame_sql_code_init)
    dataframe_sql_code_call_first_paren = dataframe_sql_code_start + text_offset
    end_paren = find_end_paren(function_code, dataframe_sql_code_call_first_paren) + 1

    dataframe_sql_code = function_code[dataframe_sql_code_start:end_paren]
    pandas_code = function_code[end_paren:]
    return dataframe_sql_code, pandas_code


def timeit(function: FunctionType):
    """
    Wrapper for measuring time based performance
    :param function:
    :return:
    """

    def timed(*args, **kw):
        ts = time.time()
        result = function(*args, **kw)  # noqa
        te = time.time()

        total_time = te - ts
        print(f"func: {function.__name__} took {total_time}")
        return total_time

    return timed


def test_performance(dataframe_sql_code: str, pandas_code: str):
    @timeit
    def dataframe_sql_time():
        exec(dataframe_sql_code)  # noqa

    @timeit
    def pandas_code_time():
        exec(pandas_code)  # noqa

    time_diff = dataframe_sql_time() - pandas_code_time()
    print(f"Time difference was {time_diff}\n")


if __name__ == "__main__":
    register_env_tables()  # noqa

    tests = get_pandas_tests()
    for test in tests[3:]:
        print(f"######### {test.__name__} #########")
        code = get_function_code(test)
        fix_code_indent(code)
        remove_assertion(code)
        code = list(filter(lambda x: x, code))
        code_string = "\n".join(code)

        code = split_into_pandas_and_dataframe_sql(code_string)
        try:
            test_performance(*code)
        except Exception as err:
            print("Code Failed")
            print("#####################")
            print("#### Your Code ####")
            print(code[0])
            print("#### Pandas Code ####")
            print(code[1])
            print("#####################")
            raise err

    remove_env_tables()  # noqa
