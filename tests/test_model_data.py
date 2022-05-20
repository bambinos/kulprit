from kulprit.data.submodel import SubModelStructure

import pytest

from . import KulpritTest


class TestData(KulpritTest):
    """Test the methods used for building ModelData objects."""

    def test_has_intercept(self, ref_model):
        assert ref_model.data.structure.has_intercept

    def test_intercept_only_submodel(self, ref_model):
        # build restricted model object
        structure_factory = SubModelStructure(ref_model.data)
        sub_model_structure = structure_factory.create([])

        assert sub_model_structure.X.shape == (
            ref_model.data.structure.num_obs,
            1,
        )

    def test_build_submodel(self, ref_model):
        # build restricted model object
        structure_factory = SubModelStructure(ref_model.data)
        sub_model_structure = structure_factory.create(["x"])

        assert sub_model_structure.X.shape == (ref_model.data.structure.num_obs, 2)
        assert sub_model_structure.model_size == 1

    def test_build_negative_size_submodel(self, ref_model):
        with pytest.raises(UserWarning):
            ref_model.project(terms=-1)

    def test_build_too_large_submodel(self, ref_model):
        with pytest.raises(UserWarning):
            ref_model.project(terms=-1)
