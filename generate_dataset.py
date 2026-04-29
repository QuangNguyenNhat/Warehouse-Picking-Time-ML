import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
rows = 10_000

distance = rng.uniform(5, 40, rows)
load = rng.uniform(10, 100, rows)
congestion = rng.integers(0, 6, rows)
shelf_level = rng.integers(1, 6, rows)
fragility = rng.uniform(0, 1, rows)
noise = rng.normal(0, 5, rows)

picking_time = (
    10
    + 1.8 * distance
    + 0.3 * load
    + 5.0 * congestion
    + 2.0 * shelf_level
    + 4.0 * fragility
    + 0.02 * distance * load
    + 3 * np.sin(distance / 10)
    + noise
)

df = pd.DataFrame(
    {
        "distance": distance,
        "load": load,
        "congestion": congestion,
        "shelf_level": shelf_level,
        "fragility": fragility,
        "picking_time": picking_time,
    }
)

out_path = Path(r"C:\Users\shinb\eai-project-laptop\src\nhatnguyen_gradproject\Warehouse_Picking_Time_Project\data\picking_time_data.csv")
preview_path = Path(r"C:\Users\shinb\eai-project-laptop\src\nhatnguyen_gradproject\Warehouse_Picking_Time_Project\Outputs\Task1_dataset_preview.csv")

out_path.parent.mkdir(parents=True, exist_ok=True)
preview_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(out_path, index=False)
df.head(50).to_csv(preview_path, index=False)

print(df.shape)
print(df.head())
