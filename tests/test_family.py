from kulprit.families.family import Family

import pytest

from tests import KulpritTest


class MyFamily:
    def __init__(self):
        self.name = "bad-family"


class MyModel:
    def __init__(self):
        self.family = MyFamily()


class TestFamily(KulpritTest):
    """Test the family module used in the procedure."""

    def test_not_implemented_family(self):
        """Test that unimplemented families raise a warning."""

        with pytest.raises(NotImplementedError):
            bad_model = MyModel()
            Family(model=bad_model)
