import pandas as pd
import json
import os

# Load the clean oracle dataset
df = pd.read_csv('../synthetic/oracle_dataset.csv')  

# Group data by user
user_data = []
for student_id, user_df in df.groupby('studentid'):
    q_ids = user_df['itemid'].tolist()
    labels = user_df['correct'].tolist()
    difficulties = user_df['item_difficulty'].tolist()
    abilities = user_df['student_skill_level'].tolist()

    # Create data entry in the format expected by BOBCAT with item difficulties and student_abilities 
    entry = {
        'user_id': int(student_id),
        'q_ids': q_ids,
        'labels': labels,
        'item_difficulties': difficulties,
        'student_abilities': abilities
    }
    user_data.append(entry)

# Save to JSON
os.makedirs('oracle', exist_ok=True)
with open('oracle/oracle_dataset.json', 'w') as f:
    json.dump(user_data, f)

print(f"Converted {len(user_data)} students to JSON format.")
