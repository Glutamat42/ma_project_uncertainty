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


def parse_args():
    parser = argparse.ArgumentParser('')
    # @formatter:off
    parser.add_argument('--source_dir', required=True, type=str)
    parser.add_argument('--target_dir', required=True, type=str)
    parser.add_argument('--source_files', nargs='+', required=True, type=str, help='paths to source files, not relative to --source_dir')
    parser.add_argument('--test_split', default=20, type=int)
    parser.add_argument('--valid_split', default=10, type=int)
    parser.add_argument('--min_speed', default=5, type=int)
    parser.add_argument('--remove_crossings', default=True, type=utils.bool_flag)
    parser.add_argument('--min_distance_to_crossing', default=30, type=int)
    parser.add_argument('--timeshift', default=2, type=int, help='timeshift in frames (10 Hz) for labels, only reducing is supported (increasing might work; untested')
    parser.add_argument('--steering_adjustment', default=5.406788214535221, type=float, help='adjust error of steering wheel angels')
    parser.add_argument('--target_resolution', type=int, nargs='+', default=(576, 324), help='width, height')
    parser.add_argument('--target_resolution_2nd', type=int, nargs='+', default=None, help='set to enable a second output resolution, will result in exporting two frames per source frame')
    parser.add_argument('--ignore_direction', type=utils.bool_flag, default=True, help='balancing: ignore direction (left/right) and balance only base on angle - like balancing by abs(angle)')
    parser.add_argument('--num_workers', type=int, default=0, help='threads for image resize, 0 to disable multithreading')
    parser.add_argument('--max_steering_angle', type=int, default=360, help='samples with abs(canSteering) > max_steering_angle will be dropped')
    parser.add_argument('--bin_cut', type=int, default=None, help='max number of items per bin, if None: size of smallest bin')
    # @formatter:on

    return parser.parse_args()


def main(args):
    plt.style.use(['science', 'ieee'])

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
    plt.xlabel("Steering wheel angle")
    plt.ylabel("Number of samples")
    if args.ignore_direction:
        plt.xlim(left=0)
    plt.ylim(bottom=1)
    plt.gca().set_aspect(1. / plt.gca().get_data_ratio())
    # plt.title("Drive360")
    plt.savefig(os.path.join(args.target_dir, 'source_data_distribution_360.png'))
    plt.close()
    df.canSteering.add(args.steering_adjustment).abs().hist(bins=int(args.max_steering_angle / 2), range=(0, args.max_steering_angle),
                                                            log=True)
    plt.xlabel("Steering wheel angle")
    plt.ylabel("Number of samples")
    if args.ignore_direction:
        plt.xlim(left=0)
    plt.ylim(bottom=1)
    plt.gca().set_aspect(1. / plt.gca().get_data_ratio())
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
    plt.xlabel("Steering wheel angle")
    plt.ylabel("Number of samples")
    plt.gca().set_aspect(1. / plt.gca().get_data_ratio())
    if args.ignore_direction:
        plt.xlim(left=0)
    plt.savefig(os.path.join(args.target_dir, 'unbalanced_data_distribution_with_angles_above_max_steering_angle.png'))
    plt.close()
    df.canSteering.abs().hist(bins=int(150 / 2), range=(0, 150), log=True)
    plt.xlabel("Steering wheel angle")
    plt.ylabel("Number of samples")
    if args.ignore_direction:
        plt.xlim(left=0)
    plt.gca().set_aspect(1. / plt.gca().get_data_ratio())
    # plt.title("After preprocessing")
    plt.savefig(os.path.join(args.target_dir, 'unbalanced_data_distribution_with_angles_till_150.png'))
    plt.close()
    df = df[abs(df.canSteering) <= args.max_steering_angle]
    log.info(f'filter too high steerings: {len(df)} ({int(len(df) / src_row_count * 100)}%)')

    # balance
    df.canSteering.abs().hist(bins=int(args.max_steering_angle / 2), range=(0, args.max_steering_angle), log=True)
    plt.xlabel("Steering wheel angle")
    plt.ylabel("Number of samples")
    if args.ignore_direction:
        plt.xlim(left=0)
    plt.ylim(bottom=1)
    plt.gca().set_aspect(1. / plt.gca().get_data_ratio())
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
    test_sample_count = int(len(df) / 100 * args.test_split)
    valid_sample_count = int(len(df) / 100 * args.valid_split)
    shannon_entropy = scipy.stats.entropy(list(df['bins_canSteering'].value_counts()))
    balance = shannon_entropy / np.log(len(labels))
    log.info(f"balance: {balance} "
             f"entropy: {shannon_entropy} "
             f"total frames: {len(df)} "
             f"min_items: {min_items} "
             f"max_steering_angle: {args.max_steering_angle}")
    # normalize to [0,1]
    steering_mean = -args.max_steering_angle
    steering_std = 2 * args.max_steering_angle
    # standardize mean=0 std=1
    # steering_mean = float(df['canSteering'].mean())
    # steering_std = float(df['canSteering'].std())

    meta = {
        'dataset_type': 'driving',
        'label': {
            'mean': steering_mean,
            'std': steering_std
        },
        # 'canSpeed': {
        #     'mean': float(df['canSpeed'].mean()),
        #     'std': float(df['canSpeed'].std())
        # },
        'len': {
            'train': len(df) - test_sample_count - valid_sample_count,
            'test': test_sample_count,
            'valid': valid_sample_count
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
        df.canSteering.abs().hist(bins=round(args.max_steering_angle/2), range=(0, args.max_steering_angle))
        plt.xlim(left=0)
    else:
        df.canSteering.abs().hist(bins=args.max_steering_angle, range=(-args.max_steering_angle, args.max_steering_angle))
    plt.xlabel("Steering wheel angle")
    plt.ylabel("Number of samples")
    plt.gca().set_aspect(1. / plt.gca().get_data_ratio())
    # plt.title("Final dataset")
    plt.savefig(os.path.join(args.target_dir, 'data_distribution.png'))
    # sys.exit(0)

    # normalize canSpeed and canSteering
    # df['canSpeed'] = (df['canSpeed'] - df['canSpeed'].mean()) / df['canSpeed'].std()
    df['canSteering'] = (df['canSteering'] - steering_mean) / steering_std

    # resize
    img_subdir_name = 'img'
    img_subdir_name_2nd = 'img_2nd_res'

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

            target_path = os.path.join(args.target_dir, img_subdir_name, row['cameraFront'])
            Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(target_path, img)

            if args.target_resolution_2nd is not None:
                img = cv2.resize(src_img, args.target_resolution_2nd)
                target_path = os.path.join(args.target_dir, img_subdir_name_2nd, row['cameraFront'])
                Path(os.path.dirname(target_path)).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(target_path, img)

    def fill_queue(data_queue, data):
        """ fill data_queue with the elements of data """
        for index, row in data.iterrows():
            data_queue.put(row, block=True)
        print(end="\r")
        log.info("finished filling queue")

    def create_index_files_and_safe_files(df):
        """ generate index files with different train/test/valid split algorithms and different sizes """
        # adjust path
        df['cameraFront'] = img_subdir_name + "/" + df['cameraFront']

        # split train/test and save csv
        df_only_relevant_cols = df[['bins_canSteering', 'cameraFront', 'canSteering', 'canSpeed', 'chapter']].copy().reset_index(drop=True)
        if args.target_resolution_2nd is not None:
            df_only_relevant_cols['cameraFront_2nd'] = img_subdir_name_2nd + "/" + df_only_relevant_cols['cameraFront']

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

        # one cut between test and train
        save_df(df_only_relevant_cols[:-(test_sample_count + valid_sample_count)], 'train')
        save_df(df_only_relevant_cols[-(test_sample_count + valid_sample_count):-valid_sample_count], 'test')
        save_df(df_only_relevant_cols[-valid_sample_count:], 'validation')

        # separation by chapters, train and test are closer together
        train_chapters_df = df_only_relevant_cols.copy()
        chapters = train_chapters_df['chapter'].value_counts()

        def get_and_remove_subset_by_chapters(n):
            new_df = pd.DataFrame()
            while len(new_df) < n:
                chapter = chapters.sample(n=1)
                rows = train_chapters_df.loc[train_chapters_df["chapter"] == chapter.index[0]][:n - len(new_df)]
                new_df = new_df.append(rows, ignore_index=True)
                train_chapters_df.drop(rows.index, inplace=True)
                chapters.drop(chapter.index, inplace=True)
            return new_df

        save_df(get_and_remove_subset_by_chapters(test_sample_count), 'test_chapters')
        save_df(get_and_remove_subset_by_chapters(valid_sample_count), 'validation_chapters')
        save_df(train_chapters_df, 'train_chapters')  # has to be last, because creating test and valid drops columns from train_chapters_df

        # random samples for test, train and test are very close together
        train3_df = df_only_relevant_cols.copy()
        test3_df = train3_df.sample(n=test_sample_count)
        train3_df.drop(test3_df.index, inplace=True)
        valid3_df = train3_df.sample(n=valid_sample_count)
        train3_df.drop(valid3_df.index, inplace=True)
        save_df(train3_df, 'train_random')
        save_df(test3_df, 'test_random')
        save_df(valid3_df, 'validation_random')

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
