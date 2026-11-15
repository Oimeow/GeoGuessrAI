from collections import defaultdict
import csv
from pathlib import Path
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from PIL import Image

from datasets import load_dataset



def download_countries_wsv500(
    dataset_name = "josefbednar/world-streetview-500k",
    per_country = 1000,
    target_countries = ['US', 'IN', 'BR'],
    out = None,
    image_dir = "images"
) -> Path:
    if not dataset_name:
        raise Exception("Error: no dataset name supplied!")
    
    # setup dirs
    if not out:
        out = Path(f"data/wsv500k_{per_country}")  # "data/..._3x1000" for example
    
    image_dir = out / image_dir
        
    out.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    
    # stream from dataset
    # while the dataset is 198GB, we stream by index so we can stop whenever
    stream = load_dataset(
        dataset_name,
        split='train',
        streaming=True
    )
    
    codes = target_countries  # i.e., country codes
    count_by_country = defaultdict(int)
    seen_panoids = set()  # we dont want duplicates
    
    with open(out / "manifest.csv", 'w', newline='') as f:
        # initialize
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "panoid", "country_code", "latitude", "longitude", "date", "file"
        ])
        writer.writeheader()
        
        for row in stream:
            cc = row['country_code']
            panoid = row['panoid']
            
            if cc not in codes:
                continue

            if count_by_country[cc] >= per_country:
                continue
            
            if panoid in seen_panoids:
                continue
            
            cc_dir = image_dir / cc;
            cc_dir.mkdir(exist_ok=True)
            
            fname = f"{row['image_id']}.jpg"
            fpath = cc_dir / fname
            row['image'].convert('RGB').save(fpath, quality=95)
            
            # write the object
            writer.writerow({
                'image_id': row['image_id'],
                'panoid': panoid,
                'country_code': cc,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'date': row['date'],
                'file': str(fpath)
            })
            f.flush()
            
            seen_panoids.add(panoid)
            count_by_country[cc] += 1
            
            print(dict(count_by_country))
            
            # we're done if...
            if all(count_by_country[c] >= per_country for c in codes):
                break
            
    return out  # the directory housing the dataset
    
class CountryImageDataset(Dataset):
    def __init__(self, rows: list[dict], label_to_index: dict[str, int], transform):
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["file"]) as image:
            image = image.convert("RGB")
            X = self.transform(image)

        y = self.label_to_index[row["country_code"]]
        return X, y


def get_data(
    country_codes: list[str],
    dataset_dir: Path,
    batch_size: int = 32,
    test_fraction: float = 0.2,
    seed: int = 42,
):
    manifest_path = dataset_dir / "manifest.csv"

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Keep requested countries only.
    rows = [row for row in rows if row["country_code"] in country_codes]

    # Balance the classes: retain only the count of the smallest class.
    by_country = defaultdict(list)
    for row in rows:
        by_country[row["country_code"]].append(row)

    missing = [code for code in country_codes if not by_country[code]]
    if missing:
        raise ValueError(f"No examples for: {missing}")

    smallest_class_size = min(len(by_country[code]) for code in country_codes)

    rng = random.Random(seed)
    train_rows = []
    test_rows = []

    # Split within each country, so train and test have equal class balance.
    for code in country_codes:
        country_rows = by_country[code]
        rng.shuffle(country_rows)
        country_rows = country_rows[:smallest_class_size]

        test_count = max(1, int(len(country_rows) * test_fraction))

        test_rows.extend(country_rows[:test_count])
        train_rows.extend(country_rows[test_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    label_to_index = {
        code: index
        for index, code in enumerate(country_codes)
    }

    # The exact preprocessing ResNet-18 expects.
    transform = ResNet18_Weights.DEFAULT.transforms()

    train_dataset = CountryImageDataset(
        train_rows,
        label_to_index,
        transform,
    )
    test_dataset = CountryImageDataset(
        test_rows,
        label_to_index,
        transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"train={len(train_dataset)}, test={len(test_dataset)}, "
        f"per class={smallest_class_size}"
    )
    print("labels:", label_to_index)

    return train_loader, test_loader, label_to_index


if __name__ == "__main__":
    download_countries_wsv500(per_country=250, target_countries=['US', 'IN', 'BR', 'MX'])
    
