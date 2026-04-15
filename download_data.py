import os
import gdown

file_ids = {
    "LAION_small": "1wWWQxYzoQIwcz0C3b2umpVs2V97xWS-P",
}

data_path = "./data/{}.tar.gz"

for name, fid in file_ids.items():
    url = f"https://drive.google.com/uc?id={fid}"
    output = data_path.format(name)
    gdown.download(url, output, quiet=False)
    print(f"Downloaded {name} to {output}")
