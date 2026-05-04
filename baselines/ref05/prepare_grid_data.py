

import argparse
import os

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def write_tfrecord_gz(filepath, records):


    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(filepath, options=options) as writer:
        for record in records:
            feature = {k: _int64_feature(v) for k, v in record.items()}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())


def prepare_items(n_items, output_dir):

    items_dir = os.path.join(output_dir, 'items')
    os.makedirs(items_dir, exist_ok=True)

    records = [{'id': [item_id]} for item_id in range(n_items)]

    filepath = os.path.join(items_dir, 'items.tfrecord.gz')
    write_tfrecord_gz(filepath, records)
    print(f"  items: {len(records)} records -> {filepath}")


def prepare_sequences(split_name, history_dict, label_dict, output_dir, sub_dir,
                      max_seq_len=120, min_hist_len=2):


    seq_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(seq_dir, exist_ok=True)

    records = []

    if split_name == 'training':
        for user_id, items in history_dict.items():
            items = [int(x) for x in items]
            if len(items) < min_hist_len:
                continue
            records.append({
                'sequence_data': items[-max_seq_len:],
                'user_id': [int(user_id)]
            })
    else:
        for user_id, labels in label_dict.items():
            labels = [int(x) for x in labels]
            if len(labels) == 0:
                continue
            history = [int(x) for x in history_dict.get(user_id, [])]
            if len(history) < 1:
                continue

            full_seq = history[-(max_seq_len - 1):] + labels[:1]
            records.append({
                'sequence_data': full_seq,
                'user_id': [int(user_id)]
            })

    filepath = os.path.join(seq_dir, f'{sub_dir}.tfrecord.gz')
    write_tfrecord_gz(filepath, records)
    print(f"  {sub_dir}: {len(records)} records -> {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setrec_dir', required=True, help='SETRec data directory')
    parser.add_argument('--emb_path', required=True, help='Path to .npy embedding file')
    parser.add_argument('--output_dir', required=True, help='Output directory for GRID data')
    parser.add_argument('--max_seq_len', type=int, default=120, help='Max sequence length')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    print("Loading SETRec splits...")
    train_dict = np.load(os.path.join(args.setrec_dir, 'training_dict.npy'), allow_pickle=True).item()
    val_dict = np.load(os.path.join(args.setrec_dir, 'validation_dict.npy'), allow_pickle=True).item()
    test_dict = np.load(os.path.join(args.setrec_dir, 'testing_dict.npy'), allow_pickle=True).item()

    print(f"  train users: {len(train_dict)}, val users: {len(val_dict)}, test users: {len(test_dict)}")


    print("Converting embeddings to .pt format...")
    emb = np.load(args.emb_path)
    n_items, emb_dim = emb.shape
    print(f"  n_items={n_items}, emb_dim={emb_dim}")

    emb_pt_path = os.path.join(args.output_dir, 'embeddings.pt')
    torch.save(torch.from_numpy(emb).float(), emb_pt_path)
    print(f"  Saved: {emb_pt_path}")


    print("Preparing items TFRecord...")
    prepare_items(n_items, args.output_dir)


    print("Preparing training sequences...")
    prepare_sequences('training', train_dict, None, args.output_dir, 'training',
                      max_seq_len=args.max_seq_len)

    print("Preparing evaluation sequences...")
    prepare_sequences('evaluation', train_dict, val_dict, args.output_dir, 'evaluation',
                      max_seq_len=args.max_seq_len)


    print("Preparing testing sequences...")
    combined_train_val = {}
    all_uids = set(list(train_dict.keys()) + list(val_dict.keys()))
    for uid in all_uids:
        train_items = list(train_dict.get(uid, []))
        val_items = list(val_dict.get(uid, []))
        combined_train_val[uid] = train_items + val_items
    prepare_sequences('testing', combined_train_val, test_dict, args.output_dir, 'testing',
                      max_seq_len=args.max_seq_len)

    print("\nDone! Output directory structure:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            fpath = os.path.join(root, file)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f'{subindent}{file} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
