import os
import logging
import pathlib
import sys
import tensorflow as tf


def time_display(s):
    d = s // (3600*24)
    s -= d * (3600*24)
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    str_time = "{:1d}d ".format(int(d)) if d else " "
    return str_time + "{:0>2d}:{:0>2d}:{:0>2d}".format(int(h), int(m), int(s))

LOGS_FNAME = 'experiment.log'
MODEL_DIR = './models'
METRICS_DIR = './metrics'


def set_up_experiment_logging(
        experiment_name,
        log_fpath,
        model_snapshot_dir,
        metrics_dir,
        is_debug=False,
        stdout=False,
        clear_old_data=False):
    # log
    log_root_path = os.path.dirname(log_fpath)
    if not os.path.exists(log_root_path):
        os.makedirs(log_root_path)
    if os.path.exists(log_fpath) and clear_old_data:
        # clear old logs, TODO might not want
        os.remove(log_fpath)

    logger = get_logger(experiment_name, log_fpath, stdout=stdout)

    # tensorflow summary
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    tf_summary_writer = (
        tf.summary.create_file_writer(metrics_dir))

    # model path
    if not os.path.exists(model_snapshot_dir):
        os.makedirs(model_snapshot_dir)
    if clear_old_data:
        for f in os.listdir(model_snapshot_dir):
            os.remove(os.path.join(model_snapshot_dir, f))
    return logger, tf_summary_writer, model_snapshot_dir


def get_logger(experiment_name, filename, stdout=False):
    """
    Return a logger instance to a file
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    # logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    if stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)
    return logger
