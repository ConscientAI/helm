import os
from typing import Dict, List

import datasets

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_directory_exists
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    TRAIN_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)


class GlobalMMLUScenario(Scenario):
    """Global MMLU is a multilingual evaluation set spanning 42 languages, including English. This dataset combines
    machine translations for MMLU questions along with professional translations and crowd-sourced post-edits. It also
    includes cultural sensitivity annotations for a subset of the questions (2850 questions per language) and
    classifies them as Culturally Sensitive (CS) or Culturally Agnostic (CA). These annotations were collected as part
    of an open science initiative led by Cohere Labs in collaboration with many external collaborators from both
    industry and academia.

    - https://huggingface.co/datasets/CohereLabs/Global-MMLU
    - https://aclanthology.org/2025.acl-long.919/
    """

    name = "global_mmlu"
    description = "Global MMLU"
    tags = ["knowledge", "multiple_choice"]

    # Map the dataset's 'dev' and 'test' splits to HELM's train and test splits.
    HF_SPLIT_TO_HELM_SPLIT = {"dev": TRAIN_SPLIT, "test": TEST_SPLIT}
    # Map the dataset's answer option column names (e.g., option_a) to the corresponding answer letters in the 'answer'
    # column (e.g., A).
    OPTION_MAPPING = {"option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D"}
    SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def __init__(self, locale: str, subject: str):
        super().__init__()
        """Initializes the Global MMLU scenario.

        :param locale: Scenario language
        :param subject: Scenario subject
        """
        # The 'locale' value is validated when the dataset is loaded from Hugging Face. The 'subject' value is
        # validated explicitly to avoid silently returning zero rows.
        if subject not in self.SUBJECTS:
            raise ValueError(f"Unsupported subject '{subject}'. Supported subjects: {', '.join(self.SUBJECTS)}")
        self.locale: str = locale
        self.subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        """Generates instances for the given locale and subject by loading the Global MMLU dataset from Hugging Face.

        :param output_path: Output path
        :return: Instances list
        """
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
        # Pin a specific dataset revision to ensure the same data is loaded across runs.
        dataset_splits: Dict[str, datasets.Dataset] = datasets.load_dataset(
            "CohereLabs/Global-MMLU",
            self.locale,
            revision="0e619dbeb34206cd48705a1a0ea7fb21cae09993",
            cache_dir=cache_dir,
        )
        instances: List[Instance] = []
        for split_name, dataset in dataset_splits.items():
            assert isinstance(dataset, datasets.Dataset)
            for row_index, row in enumerate(dataset):
                # Select rows with the given subject because each dataset split contains rows (questions) for all
                # subjects.
                if self.subject == row["subject"]:
                    input = Input(text=row["question"])
                    references: List[Reference] = []
                    # Dataset format is given below:
                    #
                    # | question | option_a | option_b | option_c | option_d | answer |
                    # |----------|----------|----------|----------|----------|--------|
                    # | Find the |     0    |     4    |     2    |     6    |    B   |
                    # | degree.. |          |          |          |          |        |
                    #
                    # Question -> Instance object
                    # Question answer -> Reference object
                    for option in self.OPTION_MAPPING:
                        references.append(
                            Reference(
                                # Each answer option value is set as the 'output' (e.g., 0) of a 'Reference' object.
                                output=Output(text=str(row[option])),
                                # Assign the correct tag if answer option column name (e.g., option_b) corresponds to
                                # the 'answer' column value (e.g., B).
                                tags=[CORRECT_TAG] if self.OPTION_MAPPING[option] == row["answer"] else [],
                            )
                        )
                    instance = Instance(
                        id=f"id{row_index}",
                        input=input,
                        references=references,
                        split=self.HF_SPLIT_TO_HELM_SPLIT[split_name],
                    )
                    instances.append(instance)
                else:
                    # Skip rows that do not have the given subject.
                    continue

        return instances

    def get_metadata(self) -> ScenarioMetadata:
        """Generates the scenario metadata.

        :return: Scenario metadata
        """
        return ScenarioMetadata(
            name="global_mmlu",
            display_name="Global MMLU",
            description="A multilingual extension of the MMLU benchmark covering 42 languages with cultural sensitivity annotations ([Singh et al., 2024](https://arxiv.org/abs/2412.03304), [dataset](https://huggingface.co/datasets/CohereLabs/Global-MMLU))",
            taxonomy=TaxonomyInfo(
                task="multiple-choice question answering",
                what="MMLU benchmark translated into 42 languages",
                when="2024",
                who="Cohere Labs community and professional annotators",
                language=self.locale,
            ),
            main_metric="exact_match",
            main_split="test",
        )
