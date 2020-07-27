from ibis.common.exceptions import OperationNotDefinedError
import pytest

ibis_not_implemented = pytest.mark.xfail(
    raises=(OperationNotDefinedError, NotImplementedError, ValueError),
    reason="Not implemented in ibis",
)

ibis_next_bug_fix = pytest.mark.xfail(reason="Bug fixed in next ibis release")
