"""NextGen ML Lab: beginner linear regression demo.

Run:
    python src/demo_linear_regression.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "student_scores_nextgen.csv"


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    features = data[["hours_studied", "practice_tests", "attendance_percent"]]
    target = data["score"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("NextGen ML Lab - Linear Regression Demo")
    print("---------------------------------------")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")

    sample_student = pd.DataFrame(
        [{"hours_studied": 5.0, "practice_tests": 4, "attendance_percent": 82}]
    )
    predicted_score = model.predict(sample_student)[0]
    print(f"Predicted score for sample student: {predicted_score:.1f}")


if __name__ == "__main__":
    main()
