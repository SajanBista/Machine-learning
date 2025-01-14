import pandas as pd

# Step 1: Load the CSV file
file_path = '/Users/sajanbista/Desktop/MachineLearning Daily/to be done/messiGoal.csv'
messi_goals = pd.read_csv("messiGoal.csv")

# Step 2: Split the single column into multiple columns using tab (\t) as the delimiter
messi_goals_split = messi_goals['Comp.\tMatchday\tVenue\tFor\tOpponent\tResult\tPos.\tMinute\tAt score\tType of goal'].str.split('\t', expand=True)

# Step 3: Assign proper column names
messi_goals_split.columns = [
    "Competition", "Matchday", "Venue", "Team", "Opponent",
    "Opponent_Position", "Result", "Position", "Minute",
    "At_Score", "Type_of_Goal", "Extra_Info"
]

# Step 4: Drop unnecessary columns (if any)
# Example: If the "Extra_Info" column is not needed, drop it.
messi_goals_cleaned = messi_goals_split.drop(columns=["Extra_Info"], errors='ignore')

# Step 5: Display the cleaned data
print(messi_goals_cleaned.head())

# Step 6: Save the cleaned data to a new CSV file (optional)
output_file = 'cleaned_messi_goals.csv'
messi_goals_cleaned.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")

print('cleaned_messi_goals')