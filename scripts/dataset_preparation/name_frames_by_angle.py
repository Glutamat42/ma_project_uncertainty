import argparse
import json
import logging
import logging.config
import os
import queue
import sys
import time
from pathlib import Path
from time import strftime, gmtime

import numpy as np
import pandas as pd
import scipy.stats
from cv2 import cv2
from matplotlib import pyplot as plt
import multiprocessing as mp

logging.config.dictConfig({'disable_existing_loggers': True, 'version': 1})

from src.utils.common import init_logging, backup_project
from src.utils import utils

"""
Modified version of prepare_dataset.py
Output all frames to one folder with leading steering angle in filename (<steering angle><other filename stuff>
"""


def parse_args():
    parser = argparse.ArgumentParser('')
    # @formatter:off
    parser.add_argument('--source_dir', required=True, type=str)
    parser.add_argument('--target_dir', required=True, type=str)
    parser.add_argument('--source_files', nargs='+', required=True, type=str, help='paths to source files, not relative to --source_dir')
    parser.add_argument('--min_speed', default=5, type=int)
    parser.add_argument('--remove_crossings', default=True, type=utils.bool_flag)
    parser.add_argument('--min_distance_to_crossing', default=30, type=int)
    parser.add_argument('--timeshift', default=2, type=int, help='timeshift in frames (10 Hz) for labels, only reducing is supported (increasing might work; untested')
    parser.add_argument('--steering_adjustment', default=5.406788214535221, type=float, help='adjust error of steering wheel angels')
    parser.add_argument('--target_resolution', type=int, nargs='+', default=(1024, 576), help='width, height')
    parser.add_argument('--ignore_direction', type=utils.bool_flag, default=True, help='balancing: ignore direction (left/right) and balance only base on angle - like balancing by abs(angle)')
    parser.add_argument('--num_workers', type=int, default=0, help='threads for image resize, 0 to disable multithreading')
    parser.add_argument('--max_steering_angle', type=int, default=360, help='samples with abs(canSteering) > max_steering_angle will be dropped')
    parser.add_argument('--bin_cut', type=int, default=None, help='max number of items per bin, if None: size of smallest bin')
    # @formatter:on

    return parser.parse_args()


def main(args):
    # create out dir
    try:
        Path(args.target_dir).mkdir(parents=True)
    except FileExistsError:
        print(f'Output directory ({args.target_dir}) already exists - aborting')
        sys.exit(1)

    # init logging
    init_logging(args.target_dir, console_log_level=logging.DEBUG)
    log = logging.getLogger("dataset")

    # set random seed
    random_seed = 42
    np.random.seed(random_seed)

    # load data
    log.info("start loading index files")
    df = pd.DataFrame()
    for file in args.source_files:
        df = df.append(pd.read_csv(file,
                                   dtype={'cameraFront': object,
                                          'cameraRear': object,
                                          'cameraRight': object,
                                          'cameraLeft': object,
                                          'tomtom': object,
                                          'hereMmLatitude': np.float32,
                                          'hereMmLongitude': np.float32,
                                          'hereSpeedLimit': np.float32,
                                          'hereSpeedLimit_2': np.float32,
                                          'hereFreeFlowSpeed': np.float32,
                                          'hereSignal': np.float32,
                                          'hereYield': np.float32,
                                          'herePedestrian': np.float32,
                                          'hereIntersection': np.float32,
                                          'hereMmIntersection': np.float32,
                                          'hereSegmentExitHeading': np.float32,
                                          'hereSegmentEntryHeading': np.float32,
                                          'hereSegmentOthersHeading': object,
                                          'hereCurvature': np.float32,
                                          'hereCurrentHeading': np.float32,
                                          'here1mHeading': np.float32,
                                          'here5mHeading': np.float32,
                                          'here10mHeading': np.float32,
                                          'here20mHeading': np.float32,
                                          'here50mHeading': np.float32,
                                          'hereTurnNumber': np.float32,
                                          'canSpeed': np.float32,
                                          'canSteering': np.float32,
                                          'chapter': int
                                          })
                       )
    df.reset_index(drop=True, inplace=True)
    df['chapter'] = pd.Categorical(df['chapter'])

    log.info('loaded rows: ' + str(len(df)))
    src_row_count = len(df)

    ### analyze loaded dataset
    # plt.plot(df.canSteering)
    df.canSteering.add(args.steering_adjustment).abs().hist(bins=180, range=(0, 360), log=True)
    plt.savefig(os.path.join(args.target_dir, 'source_data_distribution_360.png'))
    plt.close()
    df.canSteering.add(args.steering_adjustment).abs().hist(bins=int(args.max_steering_angle / 2), range=(0, args.max_steering_angle),
                                                            log=True)
    plt.savefig(os.path.join(args.target_dir, 'source_data_distribution_max_steering_angle.png'))
    plt.close()
    # plt.show()

    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    # print(df.describe(include='all'))

    # adjust timeshift
    shiftrows = args.timeshift - 10
    shiftrows = -shiftrows

    df['canSpeed'] = df['canSpeed'].shift(shiftrows)
    df['canSteering'] = df['canSteering'].shift(shiftrows)
    chapters = df['chapter'].to_numpy()
    droplist = []
    # set start and end boundaries depending on shiftrows; essentially skip first or last rows
    # they have nan values for can* and will be deleted later
    for i in range(0 if shiftrows < 0 else shiftrows, len(df) + (shiftrows if shiftrows < 0 else 0)):
        # if df.iloc[i].chapter == df.iloc[i - shiftrows].chapter:
        #     updated_df = updated_df.append(df.iloc[i])
        if chapters[i] != chapters[i - shiftrows]:
            droplist.append(i)
    df.drop(droplist, inplace=True)

    # old code, slow but leaving it for now in case the new code causes problems
    # updated_df = pd.DataFrame()
    # chapters = df['chapter'].unique()
    # for chapter in chapters:
    #     chapter_df = df[df['chapter'] == chapter]
    #     chapter_df['canSpeed'].shift(shiftrows)
    #     chapter_df['canSteering'].shift(shiftrows)
    #     chapter_df = chapter_df.iloc[:shiftrows, :]
    #     updated_df = updated_df.append(chapter_df)
    # df = updated_df
    log.info(f'adjusted timeshift: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # filter nan
    df = df[df['chapter'].notna()]
    df = df[df['hereIntersection'].notna()]
    df = df[df['canSpeed'].notna()]
    df = df[df['canSteering'].notna()]
    log.info(f'filter nan: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # remove rows with speed below min_speed
    df = df[df.canSpeed > args.min_speed]
    log.info(f'apply speed threshold: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # steering_adjustment
    df['canSteering'] = df['canSteering'].add(args.steering_adjustment)

    # remove crossings
    if args.remove_crossings:
        df = df[df.hereIntersection > args.min_distance_to_crossing]
    log.info(f'filter intersections: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # remove rows with too high steering
    df.canSteering.abs().hist(bins=int(360 / 2), range=(0, 360), log=True)
    plt.savefig(os.path.join(args.target_dir, 'unbalanced_data_distribution_with_angles_above_max_steering_angle.png'))
    plt.close()
    df = df[abs(df.canSteering) <= args.max_steering_angle]
    log.info(f'filter too high steerings: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # balance
    df.canSteering.abs().hist(bins=int(args.max_steering_angle / 2), range=(0, args.max_steering_angle), log=True)
    plt.savefig(os.path.join(args.target_dir, 'unbalanced_data_distribution.png'))
    plt.close()
    # bins = [-1, 20, 40, 60, 1000]
    bins = [-1] + [x for x in range(2, args.max_steering_angle - 1, 2)] + [1000]
    labels = [x for x in range(len(bins) - 1)]
    if not args.ignore_direction:
        bins = [-x for x in reversed(bins[1:])] + bins[1:]
        labels = [-x for x in reversed(labels[1:])] + labels
    df['bins_canSteering'] = pd.cut(
        df['canSteering'].abs() if args.ignore_direction else df['canSteering'],
        bins=bins,
        labels=labels,
        ordered=False
    )
    bin_sizes = df.groupby('bins_canSteering').size()
    log.debug(bin_sizes)

    min_items = bin_sizes.min() if args.bin_cut is None else args.bin_cut
    log.info(f"using {min_items} as min_items")
    df = df.groupby('bins_canSteering').apply(lambda g: g.sample(
        # lookup number of samples to take
        n=min_items,
        # enable replacement (oversampling) if len is less than number of samples expected
        # replace=len(g) < min_items
        # if len(g) > min_items: sample, if len(g) <= min_items and > 0: use g, else: None
    ) if len(g) > min_items else g if len(g) > 0 else None).sort_values(by=['cameraFront'])

    log.info(f'balancing: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # calculate some variables and generate meta object
    shannon_entropy = scipy.stats.entropy(list(df['bins_canSteering'].value_counts()))
    balance = shannon_entropy/np.log(len(labels))
    log.info(f"balance: {balance} entropy: {shannon_entropy} total frames: {len(df)} min_items: {min_items} max_steering_angle: {args.max_steering_angle}")
    meta = {
        'canSteering': {
            'mean': float(df['canSteering'].mean()),
            'std': float(df['canSteering'].std())
        },
        'canSpeed': {
            'mean': float(df['canSpeed'].mean()),
            'std': float(df['canSpeed'].std())
        },
        'len': {
            'total': len(df)
        },
        'balancing_options': {
            'bins': bins,
            'labels': labels
        },
        'balance': balance,
        'entropy': shannon_entropy,
        'samples_per_bin': int(min_items),
        'random_seed': random_seed,
        'created_at': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'args': vars(args)
    }

    # save plot of dataset distribution
    if args.ignore_direction:
        df.canSteering.abs().hist(bins=50, range=(0, 100))
    else:
        df.canSteering.abs().hist(bins=100, range=(-100, 100))
    plt.savefig(os.path.join(args.target_dir, 'data_distribution.png'))

    # resize
    img_subdir_name = 'img'

    # functions for multiprocessing
    def resize_frames(_, data_queue):
        """ resize frames from data_queue """
        queue_was_empty = False
        while True:
            try:
                row = data_queue.get(block=True, timeout=1)
                queue_was_empty = False
            except queue.Empty:
                if queue_was_empty:
                    log.debug("no elements left in queue")
                    return
                else:
                    queue_was_empty = True
                    continue

            src_img = cv2.imread(os.path.join(args.source_dir, row['cameraFront']))
            img = cv2.resize(src_img, args.target_resolution)

            if row['canSteering'] < 0:
                filenameprefix=f"neg_{abs(row['canSteering'])}"
            else:
                filenameprefix = f"{row['canSteering']}"
            target_path = os.path.join(args.target_dir, img_subdir_name, filenameprefix + row['cameraFront'].replace("/", "_"))
            Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(target_path, img)

    def fill_queue(data_queue, data):
        """ fill data_queue with the elements of data """
        for index, row in data.iterrows():
            data_queue.put(row, block=True)
        print(end="\r")
        log.info("finished filling queue")

    def create_index_files_and_safe_files(df):
        """ generate index files """
        # adjust path
        df['cameraFront'] = img_subdir_name + "/" + df['cameraFront']

        # save csv
        df_only_relevant_cols = df[['bins_canSteering', 'cameraFront', 'canSteering', 'canSpeed', 'chapter']].copy()

        def save_df(df, basename):
            """ safe dataframe with differently sized subsets """
            df.to_csv(os.path.join(args.target_dir, basename + ".csv"))

        save_df(df_only_relevant_cols[:], 'train')

        # save meta info
        with open(os.path.join(args.target_dir, 'meta.json'), 'w') as fp:
            json.dump(meta, fp, indent=4)

        # save source code - just to be sure it archives the whole project
        print(end="\r")
        log.info('Backup source code')
        backup_project(args.target_dir)

    # create and start filling queue
    resize_queue = mp.Queue(maxsize=len(df) + 1)
    process_fill_queue = mp.Process(target=fill_queue, args=(resize_queue, df.copy()))
    # process_fill_queue.start()  # TODO

    # create and start worker processes
    processes = [mp.Process(target=resize_frames, args=(i, resize_queue)) for i in range(args.num_workers)]
    for p in processes:
        p.start()
    log.info('resize workers started')

    process_io_stuff = mp.Process(target=create_index_files_and_safe_files, args=(df,))
    process_io_stuff.start()

    # log status and join processes
    remaining_items = resize_queue.qsize()
    while not resize_queue.empty() or process_fill_queue.is_alive():
        progress_in_pct = int((1 - resize_queue.qsize() / len(df)) * 100)
        diff = remaining_items - resize_queue.qsize()
        sys.stdout.write(
            f"\r{(str(progress_in_pct) if diff > 0 else '--').rjust(2)}%, elements remaining: {resize_queue.qsize()} ({str(diff) + 'it/s' if diff > 0 else 'queue still filling up, cant calculate speed'})")
        sys.stdout.flush()
        # print(f"{progress_in_pct if diff > 0 else '--'}%, elements remaining: {resize_queue.qsize()} ({str(diff) + 'it/s' if diff > 0 else 'queue still filling up, cant calculate speed'})")
        remaining_items = resize_queue.qsize()
        time.sleep(1)

    for p in processes:
        p.join()
    if not resize_queue.empty():
        log.error("Queue not empty. Probably queue became empty before the whole dataframe was added to it. Created dataset is incomplete!")
    process_io_stuff.join()

    log.debug('finished')


if __name__ == '__main__':
    config = parse_args()
    main(config)
