# 02 — Data Preprocessing

Data preprocessing means preparing raw data before training a machine-learning model.

## Main Steps

1. Load the dataset.
2. Check missing values.
3. Fix wrong data types.
4. Encode categorical columns.
5. Scale numeric columns when needed.
6. Split data into training and testing sets.

## Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("datasets/student_scores_nextgen.csv")
X = data[["hours_studied", "practice_tests", "attendance_percent"]]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

## Key Idea

A good model starts with clean data. Bad data usually creates bad predictions.
