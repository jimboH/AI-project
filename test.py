from datasets import load_dataset
import ipdb
# 1. Define your desired sequential recommendation subset config
# Pattern: [core]_[split]_w_his_[category]
config_name = "5core_last_out_w_his_All_Beauty"

print(f"--- Fetching Sequential Dataset Split: {config_name} ---")

# 2. Fetch the dataset from the official McAuley Lab Hugging Face repository
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    config_name, 
    trust_remote_code=True
)

# 3. Inspect dataset splits (train, validation, test)
print("\nDataset Structure:")
print(dataset)
ipdb.set_trace()