import os
import shutil
import zipfile

import requests
from tqdm import tqdm


CHOSEN_CLASSES = (
    "backpack",
    "book",
    "car",
    "pizza",
    "sandwich",
    "snake",
    "sock",
    "tiger",
    "tree",
    "watermelon",
)


def file_count(path):
    count = 0
    for _, _, files in os.walk(path):
        count += len(files)
    return count


def download_file(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def sample_dataset(domain):
    domain_path = f"data/{domain}"
    train_path = f"data/{domain}_train"
    test_path = f"data/{domain}_test"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    with open(f"data/{domain}_train.txt", "r") as f:
        src_train_files = [line.split("\n")[0].split()[0] for line in f.readlines()]
    with open(f"data/{domain}_test.txt", "r") as f:
        src_test_files = [line.split("\n")[0].split()[0] for line in f.readlines()]

    for class_name in CHOSEN_CLASSES:
        class_path = os.path.join(domain_path, class_name)
        train_subset_path = os.path.join(train_path, class_name)
        test_subset_path = os.path.join(test_path, class_name)
        if not os.path.exists(train_subset_path):
            os.makedirs(train_subset_path)
        if not os.path.exists(test_subset_path):
            os.makedirs(test_subset_path)
        imgs = os.listdir(class_path)
        for img in imgs:
            img_path = os.path.join(class_path, img)
            img_path_wo_stem = "/".join(img_path.split("/")[1:])
            if img_path_wo_stem in src_train_files:
                shutil.copy(img_path, train_subset_path)
            elif img_path_wo_stem in src_test_files:
                shutil.copy(img_path, test_subset_path)
            else:
                print(
                    f"Image ({img_path}) skipped, as it doesn't belong to train nor test"
                )


if __name__ == "__main__":
    # download the data
    data_path = "data/"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    download_file(
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "data/real.zip",
    )
    download_file(
        "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
        "data/real_train.txt",
    )
    download_file(
        "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
        "data/real_test.txt",
    )
    download_file(
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
        "data/sketch.zip",
    )
    download_file(
        "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
        "data/sketch_train.txt",
    )
    download_file(
        "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt",
        "data/sketch_test.txt",
    )

    # unzip files
    print("===> Unzipping files, may require some time to complete <===")
    with zipfile.ZipFile("data/real.zip", "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting "):
            zf.extract(member, "data/")
    with zipfile.ZipFile("data/sketch.zip", "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting "):
            zf.extract(member, "data/")

    # sample the dataset
    sample_dataset("real")
    sample_dataset("sketch")

    # delete unnecessary files
    delete_files = [
        "real/",
        "sketch/",
        "real.zip",
        "sketch.zip",
        "real_train.txt",
        "real_test.txt",
        "sketch_train.txt",
        "sketch_test.txt",
    ]
    for delete_file in delete_files:
        if delete_file.endswith("/"):
            shutil.rmtree(f"data/{delete_file}")
        else:
            os.remove(f"data/{delete_file}")
    print(f"=====> Dataset download completes <=====")

    # counter number of images in the folder
    print(
        f"Domain: Real; Type: Train => Number of images: {file_count('data/real_train')}"
    )
    print(
        f"Domain: Real; Type: Test => Number of images: {file_count('data/real_test')}"
    )
    print(
        f"Domain: Sketch; Type: Train => Number of images: {file_count('data/sketch_train')}"
    )
    print(
        f"Domain: Sketch; Type: Test => Number of images: {file_count('data/sketch_test')}"
    )
