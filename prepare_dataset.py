import pandas as pd

# Load both datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels: 1 = Fake, 0 = Real
fake['label'] = 1
true['label'] = 0

# Combine into a single DataFrame
df = pd.concat([fake[['text', 'label']], true[['text', 'label']]], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Save as a new file
df.to_csv("fake_news_dataset.csv", index=False)

print("✅ Combined dataset saved as fake_news_dataset.csv")
print(f"Total samples: {len(df)} | Fake: {len(fake)} | Real: {len(true)}")
