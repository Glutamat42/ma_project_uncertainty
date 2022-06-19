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
from cv2 import cv2
from matplotlib import pyplot as plt
import multiprocessing as mp

logging.config.dictConfig({'disable_existing_loggers': True, 'version': 1})

from src.utils.common import init_logging, backup_project


def parse_args():
    parser = argparse.ArgumentParser('')
    # @formatter:off
    parser.add_argument('--source_dir', required=True, type=str)
    parser.add_argument('--target_dir', required=True, type=str)
    parser.add_argument('--source_files', nargs='+', required=True, type=str, help='paths to source files, not relative to --source_dir')
    parser.add_argument('--target_resolution', type=int, nargs='+', default=(576, 324), help='width, height')
    parser.add_argument('--num_workers', type=int, default=0, help='threads for image resize, 0 to disable multithreading')
    parser.add_argument('--blacklist_chapters', type=str, nargs='*', default=[], help='These chapters will not be part of the generated dataset')
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

    # drop blacklist_chapters
    log.info("remove blacklisted chapters")
    for chapter in args.blacklist_chapters:
        rows = df.loc[df["chapter"] == int(chapter)]
        df.drop(rows.index, inplace=True)

        # def get_and_remove_subset_by_chapters(n):
        #     new_df = pd.DataFrame()
        #     while len(new_df) < n:
        #         chapter = chapters.sample(n=1)
        #         rows = train_chapters_df.loc[train_chapters_df["chapter"] == chapter.index[0]][:n - len(new_df)]
        #         new_df = new_df.append(rows, ignore_index=True)
        #         train_chapters_df.drop(rows.index, inplace=True)
        #         chapters.drop(chapter.index, inplace=True)
        #     return new_df

    ### analyze loaded dataset
    # plt.plot(df.canSteering)
    df.canSteering.abs().hist(bins=180, range=(0, 360), log=True)
    plt.savefig(os.path.join(args.target_dir, 'data_distribution_360.png'))
    plt.close()

    # generate meta object
    meta = {
        'dataset_type': 'driving',
        'label': {
            'mean': float(df['canSteering'].mean()),
            'std': float(df['canSteering'].std())
        },
        'canSteering': {
            'mean': float(df['canSteering'].mean()),
            'std': float(df['canSteering'].std())
        },
        'canSpeed': {
            'mean': float(df['canSpeed'].mean()),
            'std': float(df['canSpeed'].std())
        },
        'random_seed': random_seed,
        'created_at': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'args': vars(args)
    }

    # resize
    img_subdir_name = 'img'

    # functions for multiprocessing
    def resize_frames(_, data_queue):
        """ resize frames from data_queue """
        queue_was_empty = 0
        while True:
            try:
                row = data_queue.get(block=True, timeout=1)
                queue_was_empty = 0
            except queue.Empty:
                if queue_was_empty == 5:
                    log.debug("no elements left in queue")
                    return
                else:
                    queue_was_empty += 1
                    continue

            src_img = cv2.imread(os.path.join(args.source_dir, row['cameraFront']))
            img = cv2.resize(src_img, args.target_resolution)

            target_path = os.path.join(args.target_dir, img_subdir_name, row['cameraFront'])
            Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def fill_queue(data_queue, data):
        """ fill data_queue with the elements of data """
        log.info("start filling queue")
        for index, row in data.iterrows():
            data_queue.put(row, block=True)
        print(end="\r")
        log.info("finished filling queue")

    def create_index_files_and_safe_files(df):
        """ generate index files with different train/test/valid split algorithms and different sizes """
        # adjust path
        df['cameraFront'] = img_subdir_name + "/" + df['cameraFront']

        # split train/test and save csv
        df_only_relevant_cols = df[['cameraFront', 'canSteering', 'canSpeed', 'chapter']].copy().reset_index(drop=True)
        df_only_relevant_cols['bins_canSteering'] = -1

        def save_df(df, basename):
            """ safe dataframe with differently sized subsets """
            df_2 = df[::2]
            df_4 = df_2[::2]
            df_8 = df_4[::2]
            df_16 = df_8[::2]
            df.to_csv(os.path.join(args.target_dir, basename + ".csv"))
            df_2.to_csv(os.path.join(args.target_dir, basename + "_1_2.csv"))
            df_4.to_csv(os.path.join(args.target_dir, basename + "_1_4.csv"))
            df_8.to_csv(os.path.join(args.target_dir, basename + "_1_8.csv"))
            df_16.to_csv(os.path.join(args.target_dir, basename + "_1_16.csv"))

        save_df(df_only_relevant_cols, 'train')

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
    process_fill_queue.start()

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
