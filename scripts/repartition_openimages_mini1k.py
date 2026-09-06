#!/usr/bin/env python3
import csv,hashlib,json,os,random,shutil
from collections import Counter
from pathlib import Path
SEED=42
CUTS={"train":(0,700),"valid":(700,850),"test":(850,1000)}
CUSTOM={"train":112,"valid":91,"test":29}
SUFFIXES={".jpg",".jpeg",".png",".bmp",".webp"}
repo=Path.cwd().resolve()
src=repo/"datasets/openimages_mini1k"
custom=repo/"datasets/laptop_mouse_yolov8_session_split"
oi=repo/"datasets/openimages_mini1k_split_700_150_150"
mixed=repo/"datasets/openimages_mini1k_session_mixed_3way"
for p in (oi,mixed):
    if p.exists(): raise FileExistsError(f"refusing to overwrite: {p}")
def images(p):
    if not p.is_dir(): raise FileNotFoundError(p)
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in SUFFIXES)
def label_for(x): return x.parent.parent/"labels"/f"{x.stem}.txt"
def link(src_path,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): raise FileExistsError(dst)
    try: os.link(src_path.resolve(),dst)
    except OSError: shutil.copy2(src_path.resolve(),dst)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()
with (src/"selected_images.csv").open(encoding="utf-8",newline="") as f:
    rows=list(csv.DictReader(f))
if len(rows)!=1000: raise RuntimeError(f"expected 1000 rows, got {len(rows)}")
random.Random(SEED).shuffle(rows)
assigned={k:[{**r,"split":k} for r in rows[a:b]] for k,(a,b) in CUTS.items()}
by_id={p.stem:p for p in images(src/"train/images")}
for split,batch in assigned.items():
    for row in batch:
        image=by_id.get(row["image_id"])
        label=src/"train/labels"/f"{row['image_id']}.txt"
        if image is None: raise FileNotFoundError(row["image_id"])
        if not label.is_file(): raise FileNotFoundError(label)
        link(image,oi/split/"images"/image.name)
        link(label,oi/split/"labels"/label.name)
oi.mkdir(parents=True,exist_ok=True)
(oi/"data.yaml").write_text(f"path: {oi}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n- laptop\n- mouse\nnc: 2\n",encoding="utf-8")
fields=list(rows[0])+["split"]
with (oi/"split_assignment.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
    for split in CUTS:w.writerows(assigned[split])
source_counts={}
for split in CUTS:
    oi_batch=images(oi/split/"images")
    custom_batch=images(custom/split/"images")
    if len(custom_batch)!=CUSTOM[split]:
        raise RuntimeError(f"custom {split}: expected {CUSTOM[split]}, got {len(custom_batch)}")
    source_counts[split]={"openimages":len(oi_batch),"custom":len(custom_batch)}
    for prefix,batch in (("oi",oi_batch),("custom",custom_batch)):
        for image in batch:
            label=label_for(image)
            if not label.is_file():raise FileNotFoundError(label)
            stem=f"{prefix}__{image.stem}"
            link(image,mixed/split/"images"/f"{stem}{image.suffix.lower()}")
            link(label,mixed/split/"labels"/f"{stem}.txt")
hashes={}
for split in CUTS:
    hashes[split]={}
    for image in images(mixed/split/"images"):
        value=sha(image)
        if value in hashes[split]:raise RuntimeError(f"duplicate inside {split}")
        hashes[split][value]=str(image)
names=list(CUTS)
for i,first in enumerate(names):
    for second in names[i+1:]:
        overlap=set(hashes[first])&set(hashes[second])
        if overlap:raise RuntimeError(f"hash leakage {first}/{second}: {len(overlap)}")
counts={k:len(v) for k,v in hashes.items()}
expected={"train":812,"valid":241,"test":179}
if counts!=expected:raise RuntimeError(f"bad counts: {counts}, expected {expected}")
(mixed/"data.yaml").write_text(f"path: {mixed}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n- laptop\n- mouse\nnc: 2\n",encoding="utf-8")
manifest={"seed":SEED,"openimages_counts":{k:len(v) for k,v in assigned.items()},"mixed_source_counts":source_counts,"mixed_counts":counts,"openimages_group_counts":{k:dict(Counter(r["group"] for r in v)) for k,v in assigned.items()},"exact_hash_leakage":0}
(mixed/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
(mixed/".ready").write_text("ready\n",encoding="utf-8")
print(json.dumps(manifest,indent=2,ensure_ascii=False))
print("DATASET READY")
