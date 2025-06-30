import numpy as np
import os
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pdb

import experiment
from superpoint.settings import EXPER_PATH

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--export_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pred_only', action='store_true')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    export_name = args.export_name if args.export_name else experiment_name
    batch_size = args.batch_size
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    assert 'eval_iter' in config

    output_dir = Path(EXPER_PATH, 'outputs/{}/'.format(export_name))
    if not output_dir.exists():
        os.makedirs(output_dir)
    checkpoint = Path(EXPER_PATH, experiment_name)
    if 'checkpoint' in config:
        checkpoint = Path(checkpoint, config['checkpoint'])

    config['model']['pred_batch_size'] = batch_size
    batch_size *= experiment.get_num_gpus()

    with experiment._init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(str(checkpoint))
        test_set = dataset.get_test_set()

        for _ in tqdm(range(config.get('skip', 0))):
            next(test_set)

        pbar = tqdm(total=config['eval_iter'] if config['eval_iter'] > 0 else None)
        i = 0
        while True:
            # Gather dataset
            data = []
            try:
                for _ in range(batch_size):
                    data.append(next(test_set))
            except (StopIteration, dataset.end_set):
                if not data:
                    break
                data += [data[-1] for _ in range(batch_size - len(data))]  # add dummy
            data = dict(zip(data[0], zip(*[d.values() for d in data])))

            # Predict
            if args.pred_only:
                p = net.predict(data, keys='pred', batch=True)
                pred = {'points': [np.array(np.where(e)).T for e in p]}
            else:
                pred = net.predict(data, keys='*', batch=True)

            # TODO colon superpoint: Filter with mask
            mask_tuple = data.get('mask', None)
            if mask_tuple is not None and 'points' in pred:
                # convert each element to numpy bool array [H, W]
                mask_tuple = [np.squeeze(np.asarray(m)).astype(bool)
                              for m in mask_tuple]

                for b in range(len(pred['points'])):
                    pts = pred['points'][b]          # (N, 2)  row-col order
                    if pts.size == 0:
                        continue

                    rows = pts[:, 0].astype(np.int32)
                    cols = pts[:, 1].astype(np.int32)
                    keep = mask_tuple[b][rows, cols]      # boolean mask

                    # keep only valid detections
                    print(f"Filtering {len(pts)} detections to {np.sum(keep)} valid detections")
                    pred['points'][b] = pts[keep]

                    if 'descriptors' in pred:
                        pred['descriptors'][b] = pred['descriptors'][b][keep]
                    if 'scores' in pred:
                        pred['scores'][b] = pred['scores'][b][keep]

            # Export
            d2l = lambda d: [dict(zip(d, e)) for e in zip(*d.values())]  # noqa: E731
            for p, d in zip(d2l(pred), d2l(data)):
                if not ('name' in d):
                    p.update(d)  # Can't get the data back from the filename --> dump
                filename = d['name'].decode('utf-8') if 'name' in d else str(i)
                filepath = Path(output_dir, '{}.npz'.format(filename))
                np.savez_compressed(filepath, **p)
                i += 1
                pbar.update(1)

            if config['eval_iter'] > 0 and i >= config['eval_iter']:
                break
