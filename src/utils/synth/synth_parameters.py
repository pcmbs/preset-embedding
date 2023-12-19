from typing import Optional, Tuple
import numpy as np


def _get_binary_params_val() -> np.ndarray:
    return tuple(np.array([0.0, 1.0], dtype=np.float32))


def _get_categorical_params_val(cardinality: int) -> np.ndarray:
    return tuple(np.linspace(0.0, 1.0, cardinality).round(3))


class SynthParameter:

    """Class holding the information of a synth parameter."""

    _frozen = False

    def __init__(
        self,
        index: int,
        name: str,
        type_: str,
        default_value: float = 0.0,
        symmetric: bool = False,
        cardinality: int = -1,
        cat_values: Optional[Tuple[float]] = None,
        cat_weights: Optional[Tuple[float]] = None,
        interval: Tuple[float] = (0.0, 1.0),
        excluded_cat_idx: Optional[Tuple[int]] = None,
    ) -> None:
        """
        Class holding the information of a synth parameter.

        Args:
        - `index` (int): The index of the synth parameter.
        - `name` (str): The name of the synth parameter.
        - `type_` (str): The type of the synth parameter. Must be one of "num", "cat", "bin".
        - `default_value` (float, optional): The default value of the synth parameter. (defaults: 0.0).
        - `symmetric` (bool, optional): Indicates if the synth parameter is symmetric. (defaults: False).
        - `cardinality` (int, optional): The cardinality of the synth parameter.
        -1 for continuous, i.e., numerical synth parameter (assuming a cardinality of 100) (defaults: -1).
        - `cat_values` (np.ndarray, optional): The categorical values of the synth parameter (only for categorical and binary synth parameters).
        Will be inferred from `cardinality` if None is given for a categorical or binary parameter. (defaults: None)
        - `interval` (Tuple[float], optional): The interval of the synth parameter (only for continuous synth parameter).
        This can be used to restrict the range of the synth parameter (defaults: (0.0, 1.0))
        - `excluded_cat_idx` (Tuple[int], optional): The excluded categorical indices of the synth parameter
        (only for categorical and binary synth parameters). Defaults to None.

        Raises:
            ValueError: If the type of the instance is unknown.

        """
        assert index > 0

        if type_ == "num":
            assert cardinality == -1
            assert cat_values is None
            assert excluded_cat_idx is None
        elif type_ == "cat":
            assert cardinality > 0
            if cat_values is None:
                cat_values = _get_categorical_params_val(cardinality)
            if excluded_cat_idx is not None:
                cat_values = tuple(np.delete(cat_values, excluded_cat_idx))
                cardinality = len(cat_values)
            if cat_weights is None:
                cat_weights = tuple(np.ones(cardinality))
            else:
                assert len(cat_weights) == cardinality
                assert np.sum(cat_weights) == 1
        elif type_ == "bin":
            cardinality = 2
            cat_values = _get_binary_params_val()
        else:
            raise ValueError(f"Unknown type: {type_}. Should be 'num', 'cat' or 'bin'")

        self.index = index
        self.name = name
        self.type = type_
        self.default_value = default_value
        self.symmetric = symmetric
        self.cardinality = cardinality
        self.cat_values = cat_values
        self.cat_weights = cat_weights
        self.interval = interval
        self.excluded_cat_idx = excluded_cat_idx

        self._frozen = True  # read-only instance

    def __setattr__(self, attr, value):
        if getattr(self, "_frozen"):
            raise AttributeError("Trying to set attribute on a frozen instance")
        return super().__setattr__(attr, value)

    def __repr__(self):
        return (
            f"SynthParameter(index={self.index}, "
            f"name={self.name}, "
            f"type={self.type}, "
            f"default_value={self.default_value}, "
            f"symmetric={self.symmetric}, "
            f"cardinality={self.cardinality}, "
            f"cat_values={list(self.cat_values)}, "
            f"interval={self.interval}, "
            f"excluded_cat_idx={self.excluded_cat_idx})"
        )


class SettingsParameter:

    """Class holding the information of a settings parameter."""

    _frozen = False

    def __init__(self, index: int, name: str, default_value: float = 0.0) -> None:
        self.index = index
        self.name = name
        self.default_value = default_value
        self.type = "none"

        self._frozen = True

    def __setattr__(self, attr, value):
        if getattr(self, "_frozen"):
            raise AttributeError("Trying to set attribute on a frozen instance")
        return super().__setattr__(attr, value)

    def __repr__(self):
        return f"SettingsParameter(index={self.index}, " f"name={self.name})"


class EmptyParameter:

    """Class holding the information of an empty parameter."""

    _frozen = False

    def __init__(self, index: int, name: str, default_value: float) -> None:
        self.index = index
        self.name = name
        self.default_value = default_value
        self.type = "none"

        self._frozen = True

    def __setattr__(self, attr, value):
        if getattr(self, "_frozen"):
            raise AttributeError("Trying to set attribute on a frozen instance")
        return super().__setattr__(attr, value)

    def __repr__(self):
        return f"EmptyParameter(index={self.index})"
