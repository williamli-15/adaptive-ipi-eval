# scripts/make_quick_subset.py  (add this tiny helper)
import json, random, pathlib
root = pathlib.Path("InjecAgent/data")
src  = {"dh": root/"test_cases_dh_base_subset.json",
        "ds": root/"test_cases_ds_base_subset.json"}
dst  = {k: root/f"test_cases_{k}_quick.json" for k in src}

random.seed(0)                           # reproducible sample
for k in src:
    data = json.load(open(src[k]))
    json.dump(random.sample(data, 10),   # ↓ 10 DH / 10 DS
              open(dst[k], "w"), indent=2)
    print("Wrote", dst[k])
