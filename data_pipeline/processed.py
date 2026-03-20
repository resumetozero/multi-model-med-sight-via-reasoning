import pandas as pd
import os

# 1. Load the data
reports = pd.read_csv('data/chestxray_IU/indiana_reports.csv')
projections = pd.read_csv('data/chestxray_IU/indiana_projections.csv')

# 2. Merge them on the 'uid' column
# This creates a row for every IMAGE, but attaches the patient's full report to it
merged_df = pd.merge(projections, reports, on='uid')

# 3. Filter for Frontal views (best for initial AI analysis)
frontal_only = merged_df[merged_df['projection'] == 'Frontal'].copy()

# 4. Create the full path to your images
image_folder = "data/chestxray_IU/images_normalized/" # Update this to your local path
frontal_only['image_path'] = frontal_only['filename'].apply(lambda x: os.path.join(image_folder, x))

# 5. Clean up: Remove cases with no findings or symptoms
df_final = frontal_only.dropna(subset=['findings', 'indication'])

print(f"Dataset ready! Total valid cases: {len(df_final)}")
print(df_final[['uid', 'indication', 'image_path']].head())