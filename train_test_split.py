from pathlib import Path
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
import shutil


men = [p for p in Path("data/men").rglob("*.jpg")]
women = [p for p in Path("data/women").rglob("*.jpg")]

shuffle(men)
shuffle(women)

print(f"found {len(men)} files for men and {len(women)} files for women!")

# ===============================================================

men_test_number = int(len(men) * 0.15)
women_test_number = int(len(women) * 0.15)

men_valid_number = int(len(men) * 0.15)
women_valid_number = int(len(women) * 0.15)

# ===============================================================

train_men = men[men_test_number+men_valid_number:]
train_women = women[women_test_number+women_valid_number:]

Path.mkdir("data/train", exist_ok=True, parents=True)
Path.mkdir("data/train/men", exist_ok=True, parents=True)
Path.mkdir("data/train/women", exist_ok=True, parents=True)

[shutil.move(p, f"data/train/men/{Path(p).name}") for p in train_men]
[shutil.move(p, f"data/train/women/{Path(p).name}") for p in train_women]

print("managing train files done")

# ===============================================================

test_men = men[:men_test_number]
test_women = women[:women_test_number]

Path.mkdir("data/test", exist_ok=True, parents=True)
Path.mkdir("data/test/men", exist_ok=True, parents=True)
Path.mkdir("data/test/women", exist_ok=True, parents=True)

[shutil.move(p, f"data/test/men/{Path(p).name}") for p in test_men]
[shutil.move(p, f"data/test/women/{Path(p).name}") for p in test_women]

print("managing test files done")

# ===============================================================

valid_men = men[men_test_number : men_test_number + men_valid_number]
valid_women = women[women_test_number : women_test_number + women_valid_number]


Path.mkdir("data/valid", exist_ok=True, parents=True)
Path.mkdir("data/valid/men", exist_ok=True, parents=True)
Path.mkdir("data/valid/women", exist_ok=True, parents=True)

[shutil.move(p, f"data/valid/men/{Path(p).name}") for p in valid_men]
[shutil.move(p, f"data/valid/women/{Path(p).name}") for p in valid_women]

print("managing valid files done")

# ===============================================================
print("operation finished!")
