from datasets import load_dataset

ds = load_dataset(
    "ds4sd/DocLayNet-v1.1", cache_dir="/media/frank/My Passport/Datasets/DocLayNet_1_1"
)
ds.save_to_disk("/media/frank/My Passport/Datasets/DocLayNet_1_1_dataset")
