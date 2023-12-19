from typing import List, Sequence, Tuple
import data.synths

from .synth_parameters import EmptyParameter, SettingsParameter


class PresetHelper:
    """
    Helper class to generate random presets for a given synthesizer.
    """

    def __init__(
        self,
        synth_name: str = "tal_noisemaker",
        params_to_exclude_str: Sequence[str] = (),
    ):
        """
        Helper class to generate random presets for a given synthesizer.

        Args
        - parameters (Sequence[Union[EmptyParameter, SettingsParameter, SynthParameter]]): synthesizer parameters
        - params_to_exclude_str (Sequence[str]): list of parameters to exclude. Can be the full name or a pattern
        that appears at the begining. In the later, the pattern must be followed by a "*". (default: ())
        - synth_name (str): name of the synthesizer
        """
        if synth_name == "tal_noisemaker":
            parameters = getattr(data.synths, synth_name).SYNTH_PARAMETERS
        else:
            raise NotImplementedError()

        self._synth_name = synth_name
        self._parameters = parameters
        self._excl_params_str = params_to_exclude_str

        self._excl_params = self._exclude_parameters(params_to_exclude_str)

        self._excl_params_idx = [p.index for p in self._excl_params]
        self._excl_params_val = [p.default_value for p in self._excl_params]

        self._used_params = [p for p in self._parameters if p.index not in self._excl_params_idx]
        self._used_params_idx = [p.index for p in self._used_params]
        self._used_params_descr = [(i, p.index, p.name, p.type) for i, p in enumerate(self._used_params)]
        self._used_num_params_idx = [p[0] for p in self._used_params_descr if p[3] == "num"]
        self._used_cat_params_idx = [p[0] for p in self._used_params_descr if p[3] == "cat"]
        self._used_bin_params_idx = [p[0] for p in self._used_params_descr if p[3] == "bin"]

        assert len(self._used_num_params_idx) + len(self._used_cat_params_idx) + len(
            self._used_bin_params_idx
        ) == len(self._used_params)

        self._grouped_used_params = self._group_params(self._used_params)
        self._grouped_cat_params = self._group_used_cat_params_per_values(self._grouped_used_params)

    @property
    def synth_name(self) -> str:
        """Return the name of the synthesizer."""
        return self._synth_name

    @property
    def excl_params_idx(self) -> List[int]:
        """Return the absolute indices of the excluded synthesizer parameters."""
        return self._excl_params_idx

    @property
    def excl_params_val(self) -> List[int]:
        """Return the default value of the excluded synthesizer parameters."""
        return self._excl_params_val

    @property
    def excl_params_str(self) -> Tuple[str]:
        """Return a tuple of the string patterns for the excluded synthesizer parameters."""
        return self._excl_params_str

    @property
    def used_params_idx(self) -> List[int]:
        """Return the absolute indices of the used synthesizer parameters."""
        return self._used_params_idx

    @property
    def num_used_params(self) -> int:
        """Return the number of used synthesizer parameters."""
        return len(self._used_params_idx)

    @property
    def used_num_params_idx(self) -> List[int]:
        """Return the indices of the numerical synthesizer parameters relative to the used parameters."""
        return self._used_num_params_idx

    @property
    def used_cat_params_idx(self) -> List[int]:
        """Return the indices of the categorical synthesizer parameters relative to the used parameters."""
        return self._used_cat_params_idx

    @property
    def used_bin_params_idx(self) -> List[int]:
        """Return the indices of the binary synthesizer parameters relative to the used parameters."""
        return self._used_bin_params_idx

    @property
    def grouped_used_params(self) -> dict:
        """
        Return the used synthesizer parameters grouped by type as dictionary.
        - The value for 'num' is a dictionary with the interval as the key and the list of the numerical
        synthesizer parameter's indices (relative to the used parameters) sharing that interval as the value.
        - The value for 'cat' is a dictionary with the list of possible values (corresponding to the
        different categories) as the key and the list of the categorical synthesizer parameter's
        indices (relative to the used parameters) sharing that list as the value.
        - The value for 'bin' is a list of the binary synthesizer parameter's indices (relative to the
        used parameters).
        """
        return self._grouped_used_params

    @property
    def grouped_used_cat_params(self) -> dict:
        """Return the used categorical synthesizer parameters grouped by category values as dictionary."""
        return self._grouped_cat_params

    @property
    def used_params_description(self) -> List[Tuple[int, str]]:
        """Return the description of the used synthesizer parameters as a
        list of tuple (<idx>, <synth-param-idx>, <synth-param-name>, <synth-param-type>)."""
        return self._used_params_descr

    def _exclude_parameters(self, pattern_to_exclude: Sequence[str]) -> list[int]:
        def match_pattern(name, pattern):
            return (name == pattern) or (pattern.endswith("*") and name.startswith(pattern[:-1]))

        return [
            p
            for p in self._parameters
            if isinstance(p, (EmptyParameter, SettingsParameter))
            or any(match_pattern(p.name, pattern) for pattern in pattern_to_exclude)
        ]

    def _group_params(self, parameters):
        grouped_params = {"num": {}, "cat": {}, "bin": []}

        for i, p in enumerate(parameters):
            if p.type == "num":
                key = p.interval
                if key in grouped_params["num"]:
                    grouped_params["num"][key].append(i)
                else:
                    grouped_params["num"][key] = [i]

            elif p.type == "cat":
                key = (p.cat_values, p.cat_weights)
                if key in grouped_params["cat"]:
                    grouped_params["cat"][key].append(i)
                else:
                    grouped_params["cat"][key] = [i]

            elif p.type == "bin":
                grouped_params["bin"].append(i)

        return grouped_params

    def _group_used_cat_params_per_values(self, grouped_used_params):
        grouped_cat_params = {}
        for (cat_values, _), indices in grouped_used_params["cat"].items():
            if cat_values in grouped_cat_params:
                grouped_cat_params[cat_values] += indices
            else:
                grouped_cat_params[cat_values] = indices

        return grouped_cat_params


if __name__ == "__main__":
    PARAMETERS_TO_EXCLUDE_STR = (
        "master_volume",
        "voices",
        "lfo_1_sync",
        "lfo_1_keytrigger",
        "lfo_2_sync",
        "lfo_2_keytrigger",
        "envelope*",
        "portamento*",
        "pitchwheel*",
        "delay*",
    )

    p_helper = PresetHelper("tal_noisemaker", PARAMETERS_TO_EXCLUDE_STR)

    rnd_sampling_info = p_helper.grouped_used_params
    print(rnd_sampling_info)
