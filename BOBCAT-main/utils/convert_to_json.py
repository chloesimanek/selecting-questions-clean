''' Noah's script for turning eedi data into JSON for BOBCAT '''

import pandas as pd
import json
import os

df = pd.read_csv('../eedi-data/train_task_3_4.csv') 

# Group data by user
user_data = []
for user_id, user_df in df.groupby('UserId'):
    q_ids = user_df['QuestionId'].tolist()
    labels = user_df['IsCorrect'].tolist() # is correct or not
    
    # Create data entry in the format expected by BOBCAT
    entry = {
        'user_id': int(user_id),
        'q_ids': q_ids,
        'labels': labels
    }
    user_data.append(entry)

# Save to JSON file 
with open('data/train_task_eedi-3.json', 'w') as f:
    json.dump(user_data, f)

print(f"Converted {len(user_data)} users to JSON format")