from __future__ import annotations
import os
import re
import mlflow
import argparse
from typing import cast
from enum import Enum


class Args(argparse.Namespace):
    identifier: str
    requested_stage: Stage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identifier", required=True)
    parser.add_argument(
        "--stage",
        dest="requested_stage",
        required=True,
        type=Stage,
        choices=list(Stage),
        help="the stage parameter is actually redundant and is used for missclick protection",
    )
    return cast(Args, parser.parse_args())


class Stage(Enum):
    """
    implements model stages with the intention of being used as aliases for additional flexibility
    """

    none = "none"
    stg = "stg"
    prd = "prd"
    archived = "archived"

    @classmethod
    def parse(cls, label: str):
        if re.match(r"\d+", label):
            return cls.none
        else:
            return cls(label)

    def next(self):
        STAGE_TRANSITIONS = {
            self.none: self.stg,
            self.stg: self.prd,
            self.prd: self.archived,
        }
        try:
            return STAGE_TRANSITIONS[self]
        except Exception:
            raise ValueError(f"Stage {self.value} cant be transitioned")


def parse_model(identifier: str) -> tuple[str, str]:
    if "/" in identifier and "@" in identifier:
        raise ValueError(f"Both '/' and '@' are in identifier = {identifier}")
    elif "/" not in identifier and "@" not in identifier:
        raise ValueError(
            "A valid model identifier must either contain a version (foo/42) or alias (foo@stg)"
        )
    name, label = re.split(r"[@/]", identifier)
    return name, label


def main():
    args = parse_args()
    name, label = parse_model(args.identifier)
    stage = Stage.parse(label)
    next_stage = stage.next()
    if next_stage != args.requested_stage:
        raise ValueError(
            "Requested stage doesnt equal the implied transition stage! "
            f"Requested = {args.requested_stage.value}, "
            f"Provided implies transition to {next_stage.value}"
        )
    client = mlflow.MlflowClient()
    if stage == Stage.none:
        model_version = client.get_model_version(name, label)
    else:
        model_version = client.get_model_version_by_alias(name, label)
    client.set_registered_model_alias(name, next_stage.value, model_version.version)
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        print(f"version={name}/{model_version.version}", file=f)


if __name__ == "__main__":
    main()
