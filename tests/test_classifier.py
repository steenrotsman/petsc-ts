from aeon.testing.estimator_checking import check_estimator

from petsc.classifier import PetsClassifier


def test_aeon_compatible_estimator():
    check_estimator(PetsClassifier(window=15))
