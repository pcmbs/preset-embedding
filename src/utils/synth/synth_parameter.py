from typing import Optional, Tuple
import numpy as np


def _get_param_val_from_card(cardinality: int) -> np.ndarray:
    return tuple(np.linspace(0.0, 1.0, cardinality).round(3))


# TODO: add cardinality to numerical parameters


class SynthParameter:
    """Class holding the information of a synth parameter."""

    _frozen = False

    def __init__(
        self,
        index: int,
        name: str,
        type_: str,
        default_value: float = 0.0,
        cardinality: int = -1,
        cat_values: Optional[Tuple[float]] = None,
        cat_weights: Optional[Tuple[float]] = None,
        interval: Tuple[float] = (0.0, 1.0),
        excluded_cat_idx: Optional[Tuple[int]] = None,
        group: Optional[str] = None,
    ) -> None:
        """
        Class holding the information of a synth parameter.

        Args:
        - `index` (int): The index of the synth parameter.
        - `name` (str): The name of the synth parameter.
        - `type_` (str): The type of the synth parameter. Must be one of "num", "cat", "bin".
        - `default_value` (float, optional): The default value of the synth parameter. (defaults: 0.0).
        - `cardinality` (int, optional): The cardinality of the synth parameter.
        -1 for continuous, i.e., numerical synth parameter (assuming a cardinality of 100) (defaults: -1).
        - `cat_values` (np.ndarray, optional): The categorical values of the synth parameter (only for categorical and binary synth parameters).
        Will be inferred from `cardinality` if None is given for a categorical or binary parameter. (defaults: None)
        - `interval` (Tuple[float], optional): The interval of the synth parameter (only for continuous synth parameter).
        This can be used to restrict the range of the synth parameter (defaults: (0.0, 1.0))
        - `excluded_cat_idx` (Tuple[int], optional): The excluded categorical indices of the synth parameter
        (only for categorical and binary synth parameters). Defaults to None.
        - `group` (str, optional): The group to which belongs the parameter (e.g., osc, vcf, vca, etc.). (Defaults: None).
        """
        assert index >= 0

        if type_ == "num" and cardinality == -1:  # numerical continuous parameters
            assert cat_values is None
            assert cat_weights is None
            assert excluded_cat_idx is None

        else:  # numerical discretized, categorical, binary parameters
            if type_ == "bin":
                cardinality = 2

            if cat_values is None:
                cat_values = _get_param_val_from_card(cardinality)

            if excluded_cat_idx is not None:
                cat_values = tuple(np.delete(cat_values, excluded_cat_idx))
                cardinality = len(cat_values)

            if cat_weights is None:
                cat_weights = tuple(np.ones(cardinality))
            else:
                assert len(cat_weights) == cardinality
                assert np.sum(cat_weights) == 1

        self.index = index
        self.name = name
        self.type = type_
        self.default_value = default_value
        self.cardinality = cardinality
        self.cat_values = cat_values
        self.cat_weights = cat_weights
        self.interval = interval
        self.excluded_cat_idx = excluded_cat_idx
        self.group = group

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
            f"cardinality={self.cardinality}, "
            f"cat_values={list(self.cat_values) if self.cat_values else 'None'}, "
            f"interval={self.interval}, "
            f"excluded_cat_idx={self.excluded_cat_idx}, "
            f"group={self.group})"
        )