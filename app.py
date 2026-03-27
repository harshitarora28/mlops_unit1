import pandas as pd

# sample dataset
data = {
    "Name": ["A", "B", "C", "D"],
    "Marks": [85, 90, 78, 92]
}

df = pd.DataFrame(data)

# basic stats
print("Dataset:\n", df)
print("\nAverage Marks:", df["Marks"].mean())
print("Max Marks:", df["Marks"].max())
print("Min Marks:", df["Marks"].min())
print("Total:", df["Marks"].sum())