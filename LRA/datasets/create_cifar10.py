import pickle
import tensorflow as tf  # TF2 eager by default
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

batch_size = 1
normalize = False

# CIFAR-10 sizes
N_TRAIN = 45000   # train[:90%]
N_VAL   = 5000    # train[90%:]
N_TEST  = 10000   # test

def decode(ex):
    x = tf.image.rgb_to_grayscale(ex['image'])  # (32,32,1), uint8
    if normalize:
        x = tf.cast(x, tf.float32) / 255.0
    else:
        x = tf.cast(x, tf.int32)
    return {'inputs': x, 'targets': ex['label']}

def build_dataset(split, take_n=None, shuffle=True, repeat=False):
    ds = tfds.load('cifar10', split=split, as_supervised=False)
    ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    if take_n is not None:
        ds = ds.take(take_n)
    return ds

# Finite datasets for serialization
train_dataset = build_dataset('train[:90%]', take_n=N_TRAIN, shuffle=True, repeat=True)  # repeat + take => finite
val_dataset   = build_dataset('train[90%:]',  take_n=N_VAL,   shuffle=False, repeat=False)
test_dataset  = build_dataset('test',         take_n=N_TEST,  shuffle=False, repeat=False)

mapping = {"train": train_dataset, "dev": val_dataset, "test": test_dataset}
sizes   = {"train": N_TRAIN, "dev": N_VAL, "test": N_TEST}

for name, ds in mapping.items():
    print(f"Building {name} ({sizes[name]} examples)...")
    ds_list = []
    for idx, inst in enumerate(ds.as_numpy_iterator()):
        # inst['inputs']: (1, 32, 32, 1) because batch_size=1
        # flatten per your original code
        ds_list.append({
            "input_ids_0": inst["inputs"][0].reshape(-1),
            "label": int(inst["targets"][0])
        })
        if (idx + 1) % 1000 == 0:
            print(f"{idx + 1}/{sizes[name]}", end="\r")

    out_path = f"./lra-image.{name}.pickle"
    print(f"\nDumping {out_path} ({len(ds_list)} records)")
    with open(out_path, "wb") as f:
        pickle.dump(ds_list, f, protocol=pickle.HIGHEST_PROTOCOL)
