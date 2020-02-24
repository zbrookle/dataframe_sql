## Execution plan


### Values of certain "random" variables 
In certaine places it was necessary to create functionality that pandas doesn't
 support and in those cases, there may be strange variables that you come across in
  the execution plan. 
  
This is what their values are:

```python
FALSE_SERIES = Series(data=[False for _ in range(0, dataframe_size)]))
NONE_SERIES = Series(data=[None for _ in range(0, dataframe_size)]))
```


# Issues that come from Pandas

- No native cross join
- No rank over(order by ...)
- No straight aggregation without groupby object
- ** No pandas date object