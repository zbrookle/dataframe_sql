# dataframe_sql

![CI](https://github.com/zbrookle/dataframe_sql/workflows/CI/badge.svg)

## Installation

```bash
pip install dataframe_sql
```

## Usage

In this simple example, a DataFrame is read in from a csv and then using the query
function you can produce a new DataFrame from the sql query. 

```python
from pandas import read_csv
from dataframe_sql import register_temp_table, query

my_table = read_csv("some_file.csv")

register_temp_table(my_table)

query(""""select * from my_table""")
```

The package currently only supports pandas but there are plans to support dask and
rapids in the future.

## Execution plan

### Values of certain "random" variables
In certain places it was necessary to create functionality that pandas doesn't support
and in those cases, there may be strange variables that you come across in the
execution plan.

This is what their values are:

```python
FALSE_SERIES = Series(data=[False for _ in range(0, dataframe_size)]))
NONE_SERIES = Series(data=[None for _ in range(0, dataframe_size)]))
```


## Issues that come from Pandas

- No native cross join
- No rank over(order by ...)
- No straight aggregation without groupby object
- ** No pandas date object