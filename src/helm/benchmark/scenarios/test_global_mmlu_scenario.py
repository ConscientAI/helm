import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.global_mmlu_scenario import GlobalMMLUScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_global_mmlu_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = GlobalMMLUScenario(locale="en", subject="abstract_algebra")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 105
        assert instances[0].input == Input(
            text="Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
        )
        assert instances[0].references == [
            Reference(output=Output(text="0"), tags=[]),
            Reference(output=Output(text="4"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="2"), tags=[]),
            Reference(output=Output(text="6"), tags=[]),
        ]
        assert instances[0].split == "test"

        scenario = GlobalMMLUScenario(locale="en", subject="anatomy")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 140
        assert instances[0].input == Input(
            text="A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
        )
        assert instances[0].references == [
            Reference(output=Output(text="paralysis of the facial muscles."), tags=[CORRECT_TAG]),
            Reference(output=Output(text="paralysis of the facial muscles and loss of taste."), tags=[]),
            Reference(output=Output(text="paralysis of the facial muscles, loss of taste and lacrimation."), tags=[]),
            Reference(
                output=Output(
                    text="paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
                ),
                tags=[],
            ),
        ]
        assert instances[0].split == "test"
