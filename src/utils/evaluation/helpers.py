import os
import io
import logging
import uncertainty_toolbox as uct

from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

from src.utils.evaluation import plots
from src.utils.evaluation.metrics import mse_equally_weighted, mse_extreme_angles, mae, rmse
from src.project_enums import EnumDatasets

log = logging.getLogger('eval_help')


def convert_normed_values_to_angles(std, mean, value, invert=True):
    value = value * std + mean
    if invert:
        return value * -1
    return value


def plt2PIL():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=250)  # main switch for image quality. Don't set dpi too high, it will result in gigantic resolutions
    buf.seek(0)
    i = Image.open(buf)
    # buf.close()
    return i


def asImg(_):
    img = plt2PIL()
    plt.close()
    return img


def generate_evaluation_plots(config, predictions, mode='save', out_name_prefix='', epoch=-2):
    plt.style.use(['science', 'ieee'])

    def save_or_show_plot(img, title):
        if mode == 'save':
            filename = f"{title}_{(out_name_prefix + '_') if len(out_name_prefix) > 0 else ''}supervised_train_ep{str(epoch + 1).zfill(3)}"
            directory = os.path.join(config.output_dir, 'plots')
            Path(directory).mkdir(exist_ok=True)
            # print(directory)
            # print(filename)
            # WebP would save ~22%, but Windows has bad support for it
            img.save(os.path.join(directory, filename) + ".png", "PNG", quality=95, optimize=True)
        else:  # 'show'
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    img_mse_over_angles = asImg(plots.plot_mse_over_angles(predictions))
    img_avg_prediction_over_angles = asImg(plots.plot_avg_prediction_over_angles(predictions))
    img_avg_prediction_over_angles_scatter = asImg(plots.plot_avg_prediction_over_angles_scatter(predictions))
    img_sample_frames = asImg(plots.plot_sample_frames(
        predictions,
        config.data_path,
        # 'driving' for call from eval_predictions.py, it's a more general "name" for the other two options
        draw_overlay=config.dataset in [EnumDatasets.drive360, EnumDatasets.driving_dataset, 'driving']))
    save_or_show_plot(img_mse_over_angles, 'mse_over_angles')
    save_or_show_plot(img_avg_prediction_over_angles, 'avg_prediction_over_angles')
    save_or_show_plot(img_avg_prediction_over_angles_scatter, 'avg_prediction_over_angles_scatter')
    save_or_show_plot(img_sample_frames, 'sample_frames')

    generated_plots = {
        'mse_over_angles': img_mse_over_angles,
        'avg_prediction_over_angles': img_avg_prediction_over_angles,
        'avg_prediction_over_angles_scatter': img_avg_prediction_over_angles_scatter,
        'sample_frames': img_sample_frames,
    }

    if 'std' in predictions:
        # calculate std in angle format if it is missing in csv file. Not used here currently so disabled for performance reasons
        # if 'std_as_angle' not in predictions:
        #     predictions['std_as_angle'] = predictions["std"] * config.meta['label']['std']
        uct.viz.plot_intervals_ordered(predictions['prediction'].values,
                                       predictions['std'].values,
                                       predictions['target'].values,
                                       n_subset=50,
                                       num_stds_confidence_bound=2)
        plt.gcf().set_size_inches(3.3, 2.5)
        plt.style.use(['science', 'ieee'])
        img_intervals_ordered = asImg(None)
        save_or_show_plot(img_intervals_ordered, 'img_intervals_ordered')
        generated_plots['intervals_ordered'] = img_intervals_ordered

        uct.viz.plot_calibration(predictions['prediction'].values, predictions['std'].values, predictions['target'].values)
        plt.gcf().set_size_inches(3.3, 2.5)
        plt.style.use(['science', 'ieee'])
        img_calibration = asImg(None)
        save_or_show_plot(img_calibration, 'img_calibration')
        generated_plots['calibration'] = img_calibration

    return generated_plots


def generate_evaluation_metrics(predictions, loss):
    metrics = {
        'validation_mse': loss,
        'validation_mse_equally_weighted': mse_equally_weighted(predictions),
        'validation_mse_extreme_values': mse_extreme_angles(predictions),
        'validation_mae': mae(predictions),
        'validation_rmse': rmse(predictions)
    }

    if 'std' in predictions:
        # calculate std in angle format if it is missing in csv file. Not used here currently so disabled for performance reasons
        # if 'std_as_angle' not in predictions:
        #     predictions['std_as_angle'] =  predictions["std"] * config.meta['label']['std']

        # log all uct metrics. Takes some time (but not too much) so I disabled it
        # uct_metrics = uct.get_all_metrics(predictions['prediction'].values, predictions['std'].values, predictions['target'].values)
        # log.info(uct_metrics)

        """ all options:    
               uct_metrics = uct.get_all_metrics(...)
               [f"{k}: {v.keys()}" for k,v in uct_metrics.items()]

               ["accuracy: dict_keys(['mae', 'rmse', 'mdae', 'marpd', 'r2', 'corr'])",
                "avg_calibration: dict_keys(['rms_cal', 'ma_cal', 'miscal_area'])",
                "adv_group_calibration: dict_keys(['ma_adv_group_cal', 'rms_adv_group_cal'])",
                "sharpness: dict_keys(['sharp'])",
                "scoring_rule: dict_keys(['nll', 'crps', 'check', 'interval'])"] """
        metrics['uncertainty'] = {
            # basically the same as ma_cal as per docs miscalibration_area()
            'miscal_area': uct.miscalibration_area(predictions['prediction'].values,
                                                   predictions['std'].values,
                                                   predictions['target'].values),
            'nll': uct.nll_gaussian(predictions['prediction'].values, predictions['std'].values, predictions['target'].values),
            'crps': uct.crps_gaussian(predictions['prediction'].values, predictions['std'].values, predictions['target'].values),
            'sharpness': uct.sharpness(predictions['std'].values),
        }

    return metrics


def generate_evaluations(config, predictions, loss, mode='save'):
    metrics = generate_evaluation_metrics(predictions, loss),
    generated_plots = generate_evaluation_plots(config, predictions, mode)
    return metrics, generated_plots
