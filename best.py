import pandas as pd

'''
Calculate the theoretical best prediction accuracy.
'''

'''
    Notes: 
        - mean absolute error (MAE) is the average of the absolute differences between predicted and actual values
        - fp_fn_rate = (correct.round() - probability).abs().mean()
        - e.g. correct = 1 -> 1 --- (1-0.8) = 0.2 ****** CORRECT is already 1 or 0 
'''

df = pd.read_csv("data/synthetic/noise_df.csv")


theoretical_best = (df['probability'].round() == df['correct']).mean()

'''
    Example:
        - probability = 0.8, 0.2
        - rounded_probability = 1, 0
        - correct = 1, 1
        - [1,0] == [1,1] -> 0.5
        
'''

# Print result
print(f"Theoretical best prediction: {theoretical_best:f}")
