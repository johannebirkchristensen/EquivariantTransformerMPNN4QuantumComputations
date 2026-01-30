from data_loader_qm9_v2 import get_qm9_loaders

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adds project root

from configs.QM9.config import config
db_path = config.get('db_path', 'qm9.db')
BATCH_SIZE = config.get('batch_size', 16)
train_loader, val_loader, test_loader = get_qm9_loaders(db_path, batch_size=BATCH_SIZE)
print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")



dataset =train_loader.dataset
for i in range(len(dataset)):
    try:
        _ = dataset[i]
    except Exception as e:
        print("Failed at i=", i, "key=", dataset.keys[i], "error=", repr(e))
        break
print("checked", i+1, "samples")
