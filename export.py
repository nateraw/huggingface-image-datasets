import pickle
from pathlib import Path

import tensorflow as tf

root = Path('./PetImages')
num_skipped = 0
examples = []
for folder_name in ("Cat", "Dog"):
    for fpath in (root / folder_name).glob('*'):
        with fpath.open('rb') as f:
            is_jfif = tf.compat.as_bytes('JFIF') in f.peek(10)
            if is_jfif:
                examples.append({'img_bytes': f.read(), 'labels': folder_name.lower()})
                continue
        
        num_skipped+=1
        fpath.unlink()

print("Deleted %d images" % num_skipped)

with Path("train.pt").open('wb') as f:
    pickle.dump(examples, f)
