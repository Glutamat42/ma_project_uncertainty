import math
import operator
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cv2 import cv2


def draw_values_on_frame(frame: np.array, label, prediction, std=None):
    """ visualize label and prediction on frame """
    if not isinstance(frame, np.ndarray):
        frame = frame.numpy().transpose(1, 2, 0).copy()
    else:
        frame = frame.copy()

    gt_color = (0, 0, 255)
    pred_color = (255, 0, 0)
    pred_color_uct = (255, 127, 0)

    height = frame.shape[0]
    width = frame.shape[1]

    # predictions
    diagram_radius = 75
    diagram_anchor = (diagram_radius + 5, height - 5)

    # draw uncertainty if available
    # do this first so it is in the background
    if std is not None:
        # calculate sample count and alpha based on uncertainty value (more samples if uncertainty is high)
        std_samples = max(min(math.ceil(std) * 100, 750), 25)  # max 500, min 1 samples
        alpha = 1 / (std_samples / 75)  # Transparency factor. change last number to modify transparency. Lower means more transparent

        uct_samples = np.array([np.random.normal(prediction, std) for x in range(std_samples)])
        for sample in uct_samples:
            prediction_offset = (round(diagram_radius * math.cos((sample - 90) * math.pi / 180.0)),
                                 round(diagram_radius * math.sin((sample - 90) * math.pi / 180.0)))

            # cv2 does not support alpha (at least for lines), so this workaround is required
            # - https://answers.opencv.org/question/210898/how-to-draw-line-with-alpha-transparency/
            # - https://stackoverflow.com/questions/69432439/how-to-add-transparency-to-a-line-with-opencv-python
            # - https://gist.github.com/IAmSuyogJadhav/305bfd9a0605a4c096383408bee7fd5c
            overlay = frame.copy()

            cv2.line(overlay, diagram_anchor, tuple(map(operator.add, diagram_anchor, prediction_offset)), pred_color_uct, 7)

            # apply overlay with alpha to original frame
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # draw prediction (mean) and label
    prediction_offset = (round(diagram_radius * math.cos((prediction - 90) * math.pi / 180.0)),
                         round(diagram_radius * math.sin((prediction - 90) * math.pi / 180.0)))
    label_offset = (
        round(diagram_radius * math.cos((label - 90) * math.pi / 180.0)),
        round(diagram_radius * math.sin((label - 90) * math.pi / 180.0))
    )

    frame = cv2.line(frame, diagram_anchor, tuple(map(operator.add, diagram_anchor, label_offset)), gt_color, 3)
    frame = cv2.line(frame, diagram_anchor, tuple(map(operator.add, diagram_anchor, prediction_offset)), pred_color, 3)

    # legend
    def _draw_lengend_entry(color, label, x, y):
        cv2.circle(frame, (x, y), 10, color, -1, shift=0)
        cv2.putText(frame, label, (x + 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # w_anchor = width - 155  # bottom right
    # h_anchor = height - 50
    w_anchor = 18  # top left
    h_anchor = 53
    if std is not None:
        _draw_lengend_entry(pred_color_uct, 'uncertainty', w_anchor, h_anchor - 30)
    _draw_lengend_entry(pred_color, 'prediction', w_anchor, h_anchor)
    _draw_lengend_entry(gt_color, 'label', w_anchor, h_anchor + 30)

    # cv2.imshow('',frame)
    # cv2.waitKey()

    return frame


def plot_avg_prediction_over_angles_scatter(data, hexamode=False):
    """

    Args:
        data: contains "prediction" and "target"

    Returns:

    """
    df = data.copy()
    min_val, max_val, bin_count = _get_value_ranges(df)

    if hexamode:
        plt.hexbin(df['target_as_angle'], df['prediction_as_angle'], gridsize=35, cmap='binary')
    else:
        plt.scatter(df['target_as_angle'], df['prediction_as_angle'], s=0.1)

    # set value range
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # set aspect ratio
    plt.gca().set_aspect('equal')

    # y = x line
    xpoints = ypoints = plt.ylim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)  # y=x line


def plot_mse_over_angles(data, bin_count=180, abs_angle=False):
    """

    Args:
        data: contains "prediction" and "target"

    Returns:

    """
    df = data.copy()
    bins = np.linspace(-360, 360, bin_count + 1).astype(int).tolist()
    df['angle_bins'] = pd.cut(df['target_as_angle'].abs() if abs_angle else df['target_as_angle'],
                              bins=bins)

    gdf = df.groupby("angle_bins")

    mse_per_bin = gdf.apply(lambda x: x.squared_error.mean() if len(x) > 0 else None)
    mse_per_bin.plot(logy=False, title="mse")
    # rmse_per_bin = gdf.apply(lambda x: mean_squared_error(x.target, x.prediction) ** 0.5 if len(x) > 0 else None)
    # rmse_per_bin.plot(logy=False, title="rmse")

    # set y range for plot
    plt.ylim(0, 0.25)

    plt.xticks(rotation=25)


def _get_value_ranges(df):
    min_val = math.floor(df['target_as_angle'].min() / 10) * 10
    max_val = math.ceil(df['target_as_angle'].max() / 10) * 10
    bin_count = math.floor((max_val - min_val) / 4) + 1
    return min_val, max_val, bin_count


def plot_avg_prediction_over_angles(data, abs_angle=False):
    """

    Args:
        data: contains "prediction" and "target"

    Returns:

    """
    df = data.copy()
    min_val, max_val, bin_count = _get_value_ranges(df)
    bins = np.linspace(min_val, max_val, bin_count).astype(int).tolist()
    df['angle_bins'] = pd.cut(df['target_as_angle'].abs() if abs_angle else df['target_as_angle'],
                              bins=bins, labels=[int(x + ((bins[1] - bins[0]) / 2)) for x in bins[:-1]])

    gdf = df.groupby("angle_bins")
    avg_pred_per_bin = gdf.apply(lambda x: x.prediction_as_angle.mean())

    plt.plot(avg_pred_per_bin.axes[0].categories, avg_pred_per_bin.values)

    # set value range
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # set aspect ratio
    plt.gca().set_aspect('equal')

    # y = x line
    xpoints = ypoints = plt.ylim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)  # y=x line


def plot_sample_frames(data, dataset_dir, draw_overlay=True):
    df = data.copy()
    max_mse = df.sort_values(by='squared_error').iloc[-3:]
    df['relative_error'] = np.sqrt(df['squared_error']) / pd.concat([df['target'].abs(), df['prediction'].abs()], axis=1).min(axis=1)

    max_relative_error = df[(df['target_as_angle'].abs() > 1) & (df['prediction_as_angle'].abs() > 1)].sort_values(
        by='relative_error').iloc[-3:]
    if len(max_relative_error) < 3:
        max_relative_error = df[(df['target_as_angle'].abs() > 1)].sort_values(by='relative_error').iloc[-3:]

    random_samples = df.sample(3)

    fig = plt.figure(constrained_layout=True)
    # fig.suptitle('Example images')
    fig.set_size_inches(10, 8)
    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=3, ncols=1)

    subfigs[0].suptitle(f"max squared error\n")
    # create 1x3 subplots per subfig
    axs = subfigs[0].subplots(nrows=1, ncols=3)
    for i in range(3):
        row = max_mse.iloc[i]
        img = mpimg.imread(os.path.join(dataset_dir, row['filename']))
        img = draw_values_on_frame(img, row.target_as_angle, row.prediction_as_angle,
                                   row['std_as_angle'] if 'std_as_angle' in row else None) if draw_overlay else img
        axs[i].imshow(img)
        axs[i].set_title(f"pred: {row.prediction_as_angle : .2f} target: {row.target_as_angle : .2f} se: {row.squared_error : .2f}\n"
                         f"{row.filename[4:]}", fontsize=10)
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False)

    subfigs[1].suptitle(f"max relative error   (error / min(target, prediction); on normed values)\n")
    # create 1x3 subplots per subfig
    axs = subfigs[1].subplots(nrows=1, ncols=3)
    for i in range(3):
        row = max_relative_error.iloc[i]
        img = mpimg.imread(os.path.join(dataset_dir, row['filename']))
        img = draw_values_on_frame(img, row.target_as_angle, row.prediction_as_angle,
                                   row['std_as_angle'] if 'std_as_angle' in row else None) if draw_overlay else img
        axs[i].imshow(img)
        axs[i].set_title(f"pred: {row.prediction_as_angle : .2f} target: {row.target_as_angle : .2f} re: {row.relative_error : .2f}\n"
                         f"{row.filename[4:]}", fontsize=10)
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False)

    subfigs[2].suptitle(f"random samples\n")
    # create 1x3 subplots per subfig
    axs = subfigs[2].subplots(nrows=1, ncols=3)
    for i in range(3):
        row = random_samples.iloc[i]
        img = mpimg.imread(os.path.join(dataset_dir, row['filename']))
        img = draw_values_on_frame(img, row.target_as_angle, row.prediction_as_angle,
                                   row['std_as_angle'] if 'std_as_angle' in row else None) if draw_overlay else img
        axs[i].imshow(img)
        axs[i].set_title(f"pred: {row.prediction_as_angle : .2f} target: {row.target_as_angle : .2f} se: {row.squared_error : .2f}\n"
                         f"{row.filename[4:]}", fontsize=10)
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False)
