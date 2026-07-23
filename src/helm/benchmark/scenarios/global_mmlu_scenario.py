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
    name = "global_mmlu"
    description = "Global MMLU"
    tags = ["knowledge", "multiple_choice"]

    HF_SPLIT_TO_HELM_SPLIT = {"dev": TRAIN_SPLIT, "test": TEST_SPLIT}
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
        if subject not in self.SUBJECTS:
            raise ValueError(f"Unsupported subject '{subject}'. Supported subjects: {', '.join(self.SUBJECTS)}")
        self.locale: str = locale
        self.subject: str = subject

    def get_instances(self, output_path: str) -> List[Instance]:
        cache_dir = os.path.join(output_path, "data")
        ensure_directory_exists(cache_dir)
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
                if self.subject == row["subject"]:
                    input = Input(text=row["question"])
                    references: List[Reference] = []
                    for option in self.OPTION_MAPPING:
                        references.append(
                            Reference(
                                output=Output(text=str(row[option])),
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
                    continue

        return instances

    def get_metadata(self) -> ScenarioMetadata:
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
