import pandas as pd

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

test_df = pd.read_csv(
    "data/adult.test",
    names=COLUMNS,
    skipinitialspace=True,
    skiprows=1
)

# Convert '?' to missing values
test_df = test_df.replace("?", pd.NA)

# Remove trailing '.' from the target values
test_df["income"] = test_df["income"].str.replace(
    ".", "",
    regex=False
)

# Select 500 test records
test_df = test_df.sample(
    n=500,
    random_state=42
)

test_df.to_csv(
    "test_data.csv",
    index=False
)

print("Test data created successfully")
print("Shape:", test_df.shape)
print("\nTarget distribution:")
print(test_df["income"].value_counts())