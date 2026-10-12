import argparse
import math
import sys
from pathlib import Path

import joblib


def parse_study_hours(value: str) -> float:
    try:
        hours = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("study hours must be a number") from error

    if not math.isfinite(hours) or not 0 <= hours <= 24:
        raise argparse.ArgumentTypeError("study hours must be between 0 and 24")

    return hours


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict a student's pass probability from their study hours."
    )
    parser.add_argument(
        "study_hours",
        nargs="?",
        type=parse_study_hours,
        help="hours studied (omit to be prompted)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().with_name("student_pass_model.joblib"),
        help="path to a trained model (default: student_pass_model.joblib beside this script)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(
            f"trained model not found: {args.model}. "
            "Run train_model.py first to create it."
        )

    model = joblib.load(args.model)
    classes = list(model.classes_)
    try:
        pass_class_index = classes.index(1)
    except ValueError as error:
        raise ValueError(
            "the trained model must include class 1 for Pass"
        ) from error

    if args.study_hours is None:
        while True:
            try:
                response = input("Study hours (0-24): ")
            except EOFError:
                print("No study hours were provided.", file=sys.stderr)
                return 2

            try:
                hours = parse_study_hours(response)
                break
            except argparse.ArgumentTypeError as error:
                print(error, file=sys.stderr)
    else:
        hours = args.study_hours

    prediction = model.predict([[hours]])[0]
    pass_probability = model.predict_proba([[hours]])[0][pass_class_index]
    result = "Pass" if prediction == 1 else "Fail"
    print(
        f"Prediction for {hours:g} study hours: {result} "
        f"(pass probability: {pass_probability:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())