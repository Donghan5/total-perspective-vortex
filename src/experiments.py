from dataclasses import dataclass
from typing import Literal

LabelStrategy = Literal["event", "modality"]

SINGLE_RUN_TASKS: dict[str, tuple[int, ...]] = {
    "actual_left_vs_right_fist": (3, 7, 11),
    "imagined_left_vs_right_fist": (4, 8, 12),
    "actual_fists_vs_feet": (5, 9, 13),
    "imagined_fists_vs_feet": (6, 10, 14),
}

@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: int
    name: str
    label_strategy: LabelStrategy

    repetitions: tuple[tuple[int, ...], ...]

    def get_fold_runs(self, held_out_index: int) -> tuple[list[int], list[int]]:
        if not 0 <= held_out_index < len(self.repetitions):
            raise ValueError(
                "held_out_index must be between "
                f"0 and {len(self.repetitions) - 1}."
            )

        test_runs = list(self.repetitions[held_out_index])
        train_runs = [
            run_id for index, repetition in enumerate(self.repetitions)
            if index != held_out_index
            for run_id in repetition
        ]

        if not set(train_runs).isdisjoint(test_runs):
            raise ValueError(
                "Training and test runs must not overlap. "
            )

        return train_runs, test_runs

    def get_fold_repetitions(
            self,
            held_out_index: int,
    ) -> tuple[
        list[tuple[int, ...]],
        tuple[int, ...],
    ]:
        if not 0 <= held_out_index < len(self.repetitions):
            raise ValueError(
                "held_out_index must be between "
                f"0 and {len(self.repetitions) - 1}."
            )

        test_repetition = self.repetitions[held_out_index]

        train_repetitions = [
            repetition
            for index, repetition in enumerate(self.repetitions)
            if index != held_out_index
        ]

        return train_repetitions, test_repetition

EXPERIMENTS: dict[int, ExperimentSpec] = {
    0: ExperimentSpec(
        experiment_id=0,
        name="actual_left_vs_right_fist",
        label_strategy="event",
        repetitions=((3,), (7,), (11,)),
    ),
    1: ExperimentSpec(
        experiment_id=1,
        name="imagined_left_vs_right_fist",
        label_strategy="event",
        repetitions=((4,), (8,), (12,)),
    ),
    2: ExperimentSpec(
        experiment_id=2,
        name="actual_fists_vs_feet",
        label_strategy="event",
        repetitions=((5,), (9,), (13,)),
    ),
    3: ExperimentSpec(
        experiment_id=3,
        name="imagined_fists_vs_feet",
        label_strategy="event",
        repetitions=((6,), (10,), (14,)),
    ),
    4: ExperimentSpec(
        experiment_id=4,
        name="actual_vs_imagined_left_right",
        label_strategy="modality",
        repetitions=((3, 4), (7, 8), (11, 12)),
    ),
    5: ExperimentSpec(
        experiment_id=5,
        name="actual_vs_imagined_fists_feet",
        label_strategy="modality",
        repetitions=((5, 6), (9, 10), (13, 14)),
    ),
}

def resolve_single_run_task(test_run: int) -> tuple[str, tuple[int, ...]]:
    """ Resolving the single run tasks. """
    for task_name, run_ids in SINGLE_RUN_TASKS.items():
        if test_run in run_ids:
            return task_name, run_ids

    raise ValueError(
        f"Unknown test_run: {test_run}. "
        f"Supported runs are 3 through 14."
    )


def get_experiment(experiment_id: int) -> ExperimentSpec:
    try:
        return EXPERIMENTS[experiment_id]
    except KeyError as error:
        raise ValueError(
            f"Unknown experiment_id: {experiment_id}. "
        ) from error