from ibis.common.exceptions import OperationNotDefinedError
import pytest

ibis_not_implemented = pytest.mark.xfail(
    raises=(OperationNotDefinedError, NotImplementedError),
    reason="Not implemented in ibis",
)
