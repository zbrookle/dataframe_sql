# dataframe_sql

![CI](https://github.com/zbrookle/dataframe_sql/workflows/CI/badge.svg)
[![Downloads](https://pepy.tech/badge/dataframe-sql)](https://pepy.tech/project/dataframe-sql)
[![PyPI license](https://img.shields.io/pypi/l/dataframe_sql.svg)](https://github.com/zbrookle/dataframe_sql/blob/master/LICENSE.txt)
[![PyPI status](https://img.shields.io/pypi/status/dataframe_sql.svg)](https://pypi.python.org/pypi/dataframe_sql/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/dataframe_sql.svg)](https://pypi.python.org/pypi/dataframe_sql/)

## Installation

```bash
pip install dataframe_sql
```

## Usage

In this simple example, a DataFrame is read in from a csv and then using the query
function you can produce a new DataFrame from the sql query.

``` python
from pandas import read_csv
from dataframe_sql import register_temp_table, query

my_table = read_csv("some_file.csv")

register_temp_table(my_table, "my_table")

query("""select * from my_table""")
```

The package currently only supports pandas but there are plans to support dask and
rapids in the future.

## SQL Syntax
The SQL syntax for dataframe_sql is exactly the same as the syntax in *sql_to_ibis*,
its underlying package:

You can see the full SQL syntax [here](https://github.com/zbrookle/sql_to_ibis)

