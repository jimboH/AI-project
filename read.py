import ipdb
import pandas as pd
import json

meta_df = pd.read_json('/work/u1304848/AI/project/datasets/meta_All_Beauty.jsonl', lines=True)
# Keep only necessary columns to save memory
#meta_df = meta_df[['parent_asin', 'price', 'main_category']]

# 2. Load a subset of Reviews
review_df = pd.read_json('/work/u1304848/AI/project/datasets/All_Beauty.jsonl', lines=True)
#review_df = review_df[['parent_asin', 'rating', 'helpful_vote']]

# 3. Merge (Join) the data on 'parent_asin'
# This "correlates" the records so every review has its product's price
merged_df = pd.merge(review_df, meta_df, on='parent_asin')




ipdb.set_trace()