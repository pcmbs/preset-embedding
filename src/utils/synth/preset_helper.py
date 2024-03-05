from typing import List, Sequence, Tuple
import data.synths


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
        - params_to_exclude_str (Sequence[str]): list of parameters to exclude, i.e, parameters that are kept fixed
        during while rendering a preset and are not inputs to the preset encoder. Can be the full name or a pattern
        that appears at the {begining, end}. In the later, the pattern must be {followed, preceded} by a "*". (default: ())
        - synth_name (str): name of the synthesizer
        """
        if synth_name in ["tal_noisemaker", "dexed", "diva"]:
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

        self._relative_idx_from_name = {p.name: i for i, p in enumerate(self._used_params)}

        assert len(self._used_num_params_idx) + len(self._used_cat_params_idx) + len(
            self._used_bin_params_idx
        ) == len(self._used_params)

        self._grouped_used_params = self._group_params_for_sampling(self._used_params)

    @property
    def synth_name(self) -> str:
        """Return the name of the synthesizer."""
        return self._synth_name

    @property
    def parameters(self) -> List:
        """Return a list of SynthParameter objects of all synthesizer parameters (both used and excluded)."""
        return self._parameters

    @property
    def excl_params_idx(self) -> List[int]:
        """Return the absolute indices of the excluded synthesizer parameters."""
        return self._excl_params_idx

    @property
    def excl_params_val(self) -> List[int]:
        """Return the excluded synthesizer parameters' default value."""
        return self._excl_params_val

    @property
    def excl_params_str(self) -> Tuple[str]:
        """Return the string patterns used for excluding synthesizer parameters as tuple."""
        return self._excl_params_str

    @property
    def num_used_params(self) -> int:
        """
        Return the number of used synthesizer parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return len(self._used_params_idx)

    @property
    def used_params_idx(self) -> List[int]:
        """
        Return the absolute indices of the used synthesizer parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return self._used_params_idx

    @property
    def used_noncat_params_idx(self) -> List[int]:
        """
        Return the indices of the non categorical synthesizer parameters relative to the used parameters.

        Used synthesizer parameters refer to parameters that are allowed
        to vary across training samples and are thus inputs to the preset encoder.
        """
        return self._used_num_params_idx + self._used_bin_params_idx

    @property
    def grouped_used_params(self) -> dict:
        """
        Return a dictionary of the used synthesizer parameters grouped by types (`continuous` or `discrete`).

        Used synthesizer parameters refer to parameters that are allowed to vary across training samples
        and are thus inputs to the preset encoder.

        - The `continuous` sub-dictionary contains intervals (tuple of floats) as keys and lists of indices
        (relative to the used parameters) of numerical synthesizer parameterssharing that interval as values.
        Note: Categorical and binary synthesizer parameters are inherently discrete and are not included
        in the `continuous` sub-dictionary.

        - The `discrete` sub-dictionary itself contains three sub-dictionaries: 'num', 'cat', and 'bin'.
        Each of these sub-dictionaries has tuples (cat_values, cat_weights) as keys,
        where `cat_values` and `cat_weights` are tuples containing possible discrete values
        and associated sampling weights. Similar to the `continuous` sub-dictionary, the values are
        lists of indices (relative to the used parameters) representing discrete synthesizer parameters
        with the same possible values and weights.
        """
        return self._grouped_used_params

    @property
    def used_params_description(self) -> List[Tuple[int, str]]:
        """Return the description of the used synthesizer parameters as a
        list of tuple (<idx>, <synth-param-idx>, <synth-param-name>, <synth-param-type>)."""
        return self._used_params_descr

    def relative_idx_from_name(self, name: str) -> int:
        """
        Return the index of the synthesizer parameter relative to the used parameters given its name
        or None if not found (meaning it is either excluded or wrong)."""
        return self._relative_idx_from_name.get(name, None)

    def _exclude_parameters(self, pattern_to_exclude: Sequence[str]) -> list[int]:
        def match_pattern(name, pattern):
            return (
                (name == pattern)
                or (pattern.endswith("*") and name.startswith(pattern[:-1]))
                or (pattern.startswith("*") and name.endswith(pattern[1:]))
                or (pattern.startswith("*") and pattern.endswith("*") and name.find(pattern[1:-1]) != -1)
            )

        return [
            p
            for p in self._parameters
            if any(match_pattern(p.name, pattern) for pattern in pattern_to_exclude)
        ]

    def _group_params_for_sampling(self, parameters):
        grouped_params = {"continuous": {}, "discrete": {"num": {}, "cat": {}, "bin": {}}}

        for i, p in enumerate(parameters):
            type_key = p.type
            if p.cardinality == -1:
                key = p.interval
                if key in grouped_params["continuous"]:
                    grouped_params["continuous"][key].append(i)
                else:
                    grouped_params["continuous"][key] = [i]
            else:
                vw_key = (p.cat_values, p.cat_weights)
                if vw_key in grouped_params["discrete"][type_key]:
                    grouped_params["discrete"][type_key][vw_key].append(i)
                else:
                    grouped_params["discrete"][type_key][vw_key] = [i]

        return grouped_params


if __name__ == "__main__":
    PARAMETERS_TO_EXCLUDE_STR = (
        "main:*",
        "vc:*",
        "glob:*",
        "scop:*",
        "arp:*",
        "rvb1:*",
        "dly1:*",
        "cho2:*",
        "pha2:*",
        "rot2:*",
        "*keyfollow",
        "*velocity",
        "env1:model",
        "env2:model",
        "*trigger",
        "*release_on",
        "env1:quantise",
        "env2:quantise",
        "env1:curve",
        "env2:curve",
        "lfo1:sync",
        "lfo2:sync",
        "lfo1:restart",
        "lfo2:restart",
        "mod:rectifysource",
        "mod:invertsource",
        "mod:addsource*",
        "*revision",
        "vca:pan",
        "vca:volume",
        "vca:vca",
        "vca:panmodulation",
        "vca:panmoddepth",
        "vca:mode",
        "vca:offset",
    )

    p_helper = PresetHelper("diva", PARAMETERS_TO_EXCLUDE_STR)

    rnd_sampling_info = p_helper.grouped_used_params
    print(rnd_sampling_info)
