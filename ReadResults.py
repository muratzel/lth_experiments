import torch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

import os
import glob
import shutil
from typing import List, Dict


def plot_bv_results(experiment_folder: str, err_interval: List[float], title: str) -> None:

    bv_results = [folder for folder in glob.glob(experiment_folder+"/*") if "bv_results" in os.path.basename(folder)]

    results_dict = {"pct": [], "bias_sq": [], "var": [], "error": [], "reg": []}

    for pct_folder in bv_results:

        preds_dir = pct_folder + "/predictions"
        labels_dir = pct_folder + "/labels"

        preds_paths = glob.glob(preds_dir + "/*")
        labels_paths = glob.glob(labels_dir + "/*")

        assert len(preds_paths) == len(labels_paths)

        predictions_logits = torch.stack([torch.load(preds_path) for preds_path in preds_paths])
        predictions = torch.softmax(predictions_logits, dim=-1)

        labels = torch.stack([torch.load(labels_path) for labels_path in labels_paths])

        predictions_mean = torch.mean(predictions, dim=0, keepdim=True)

        bias_sq = torch.sum((predictions - labels[0:1]) ** 2, dim=(1, 2))
        var = torch.sum((predictions - predictions_mean) ** 2, dim=(1, 2))

        pct_results = {"bias_sq": bias_sq, "var": var, "error": var + bias_sq}

        pct = np.array(float(os.path.basename(pct_folder).split("_")[2][3:]))
        reg = np.array(str("Not Regularized" if "wo_reg" in pct_folder else "Regularized"))

        for k in ["bias_sq", "var", "error"]:
            results_dict[k].append(pct_results[k].numpy()/10000.0)

        results_dict["pct"].append(pct.repeat(results_dict["bias_sq"][-1].shape))
        results_dict["reg"].append(reg.repeat(results_dict["bias_sq"][-1].shape))

    for k, v in results_dict.items():
        results_dict[k] = np.concatenate(v)

    results_dict["Variance"] = results_dict.pop("var")
    results_dict["Bias Squared"] = results_dict.pop("bias_sq")
    results_dict["Total Error"] = results_dict.pop("error")
    results_dict["Pct. Weights Remaining"] = results_dict.pop("pct")
    results_dict["Regularized"] = results_dict.pop("reg")

    df = pd.DataFrame(data=results_dict)
    dfm = df.melt(["Regularized", "Pct. Weights Remaining"], var_name="Error Type", value_name="Error")
    dfm["Error Type"] = dfm["Regularized"] + " " + dfm["Error Type"]
    sns.lineplot(data=dfm, x="Pct. Weights Remaining", y="Error", hue="Error Type", legend="full").set(title=title)

    ax = plt.gca()
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    ax.set_ylim(err_interval)
    ax.invert_xaxis()
    plt.show()


def get_train_val_test_compute_experiment_results(experiment_folder: str, dfs_keys: List[str]) -> Dict[str, pd.DataFrame]:

    exp_result_folders = [folder for folder in glob.glob(experiment_folder + "/*")
                          if "bv_results" not in os.path.basename(folder)]

    def build_results_dict():
        return {"pct": [], "early_stop_steps": [], "test_acc": []}

    dfs_keys = set(dfs_keys)
    dfs = {}
    for k in dfs_keys:
        dfs[k] = build_results_dict()

    for exp_folder in exp_result_folders:

        exp_folder_base = os.path.basename(exp_folder)
        exp_folder_base_split = exp_folder_base.split("_")

        df_key = "_".join(exp_folder_base_split[3:])

        if df_key not in dfs_keys:
            continue

        pct = np.array(float(exp_folder_base_split[2][3:]))

        exp_runs_folders = sorted(glob.glob(exp_folder + "/lightning_logs/*"))
        exp_results = build_results_dict()

        for exp_run_folder in exp_runs_folders:

            acc = EventAccumulator(exp_run_folder)
            acc.Reload()

            exp_results["pct"].append(pct)
            exp_results["test_acc"].append(np.array(float(acc.Scalars("test_acc")[0].value)))

            val_acc_df = pd.DataFrame(acc.Scalars("val_acc"))

            max_val_acc_step = val_acc_df["step"][val_acc_df["value"] == val_acc_df["value"].max()].iloc[0]
            exp_results["early_stop_steps"].append(np.array(int(max_val_acc_step)))

        for k, v in exp_results.items():
            dfs[df_key][k].append(np.stack(v))

    empty_keys = set()
    for result_key, result_dict in dfs.items():
        for k, v in result_dict.items():
            if v:
                result_dict[k] = np.concatenate(v)
            else:
                empty_keys.add(result_key)

    for k in empty_keys:
        dfs.pop(k)

    for k, v in dfs.items():
        dfs[k] = pd.DataFrame(data=v)

    return dfs


def plot_train_val_test_compute_experiment_results(dfs_dict: Dict[str, Dict[str, pd.DataFrame]], num_runs: int,
                                                   title: str, acc_interval: List[float]) -> None:

    no_mask_means = {model_name: dfs["no_mask"].mean().to_frame().transpose()
                     for model_name, dfs in dfs_dict.items() if "no_mask" in dfs}

    for v in no_mask_means.values():
        v["pct"] = 1.0

    for model_name, dfs in dfs_dict.items():
        for mask_type, df in dfs.items():

            model_plus_mask_name = model_name + "_" + mask_type

            if mask_type != "no_mask":

                df["model_name"] = pd.Series([model_plus_mask_name] * len(df.index))

                no_mask_df = pd.concat([no_mask_means[model_name]] * num_runs, ignore_index=True)
                no_mask_df["model_name"] = pd.Series([model_plus_mask_name] * num_runs)

                dfs[mask_type] = pd.concat([df, no_mask_df], ignore_index=True)

        dfs.pop("no_mask")

    results_df = pd.concat([df for dfs in dfs_dict.values() for df in dfs.values()], ignore_index=True)
    results_df = results_df.rename(columns={"pct": "Pct. Weights Remaining", "early_stop_steps": "Early Stop Steps",
                                            "test_acc": "Test Accuracy", "model_name": "Model Name"})

    steps_df = results_df[["Pct. Weights Remaining", "Early Stop Steps", "Model Name"]]
    acc_df = results_df[["Pct. Weights Remaining", "Test Accuracy", "Model Name"]]

    sns.lineplot(data=steps_df, x="Pct. Weights Remaining", y="Early Stop Steps", hue="Model Name",
                 legend="full").set(title="Early Stop Avg Step on " + title)

    ax = plt.gca()
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    ax.invert_xaxis()
    plt.show()

    sns.lineplot(data=acc_df, x="Pct. Weights Remaining", y="Test Accuracy", hue="Model Name",
                 legend="full").set(title="Avg Accuracy on " + title)

    ax = plt.gca()
    ax.set_xscale('log', base=2)
    ax.set_ylim(acc_interval)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.2f}'))
    ax.invert_xaxis()
    plt.show()


def check_integrity(experiment_folder: str, num_runs: int, bv_results_suffix: List[str]):

    exp_result_folders = [folder for folder in glob.glob(experiment_folder + "/*")
                          if "bv_results" not in os.path.basename(folder)]

    for exp_folder in exp_result_folders:

        exp_folder_base = os.path.basename(exp_folder)
        exp_folder_base_split = exp_folder_base.split("_")

        for suffix in bv_results_suffix:

            bv_results_folder = "_".join(exp_folder_base_split[:3]) + "_bv_results" + suffix
            bv_results_folder = experiment_folder + "/" + bv_results_folder

            assert os.path.exists(bv_results_folder)
            assert os.path.exists(bv_results_folder + "/labels")
            assert len(glob.glob(bv_results_folder + "/labels/*")) == num_runs
            assert os.path.exists(bv_results_folder + "/predictions")
            assert len(glob.glob(bv_results_folder + "/predictions/*")) == num_runs
            # assert os.path.exists(bv_results_folder + "/results") or os.path.exists(bv_results_folder + "/results.zip")

        exp_runs_folders = sorted(glob.glob(exp_folder + "/lightning_logs/*"))
        assert len(exp_runs_folders) == num_runs

        for exp_run_folder in exp_runs_folders:
            acc = EventAccumulator(exp_run_folder)
            acc.Reload()
            assert len(acc.Scalars("test_acc")) == 1


def clean_folders(experiment_folder: str):

    for folder in glob.glob(experiment_folder + "/*"):

        if "bv_results" in os.path.basename(folder):

            labels_path = folder + "/labels"
            predictions_path = folder + "/predictions"

            assert os.path.exists(labels_path)
            assert os.path.exists(predictions_path)

            if os.path.exists(folder + "/results"):
                os.remove(folder + "/results")
            elif os.path.exists(folder + "/results.zip"):
                os.remove(folder + "/results.zip")

        else:

            for run_folder in glob.glob(folder + "/lightning_logs/*"):
                checkpoints_folder = run_folder + "/checkpoints"
                if os.path.exists(checkpoints_folder):
                    shutil.rmtree(checkpoints_folder)


def check_and_clean(experiment_folder: str, num_runs: int, bv_results_suffix: List[str]):
    check_integrity(experiment_folder, num_runs, bv_results_suffix)
    clean_folders(experiment_folder)


if __name__ == "__main__":

    exp_num_runs = 5

    mnist_lenet_exp = "mnist_lenet"
    cifar10_lenet_exp = "cifar10_lenet"
    cifar10_conv4_exp = "cifar10_conv4"
    cifar10_conv6_exp = "cifar10_conv6"

    # check_and_clean(mnist_lenet_exp, num_runs=exp_num_runs, bv_results_suffix=["", "_wo_reg"])
    # check_and_clean(cifar10_lenet_exp, num_runs=exp_num_runs, bv_results_suffix=["", "_wo_reg"])
    # check_and_clean(cifar10_conv4_exp, num_runs=exp_num_runs, bv_results_suffix=["", "_wo_reg"])
    # check_and_clean(cifar10_conv6_exp, num_runs=exp_num_runs, bv_results_suffix=["", "_wo_reg"])

    # plot_bv_results(mnist_lenet_exp, err_interval=[0.0, 0.25], title="LeNet Bias-Variance on MNIST")
    #

    # dfs_keys = ["no_mask", "mask", "mask_rand", "mask_wo_reg", "mask_rand_wo_reg"]

    dfs_keys = ["no_mask", "mask_wo_reg", "mask_rand_wo_reg"]

    mnist_lenet_dfs = get_train_val_test_compute_experiment_results(mnist_lenet_exp, dfs_keys)
    mnist_dfs_dict = dict(lenet=mnist_lenet_dfs)

    plot_train_val_test_compute_experiment_results(mnist_dfs_dict, num_runs=exp_num_runs, title="MNIST", acc_interval=[0.9, 1.0])

    cifar10_lenet_dfs = get_train_val_test_compute_experiment_results(cifar10_lenet_exp, dfs_keys)
    cifar10_conv4_dfs = get_train_val_test_compute_experiment_results(cifar10_conv4_exp, dfs_keys)
    cifar10_conv6_dfs = get_train_val_test_compute_experiment_results(cifar10_conv6_exp, dfs_keys)
    cifar10_dfs_dict = dict(lenet=cifar10_lenet_dfs, conv4=cifar10_conv4_dfs, conv6=cifar10_conv6_dfs)

    plot_train_val_test_compute_experiment_results(cifar10_dfs_dict, num_runs=exp_num_runs, title="CIFAR10",
                                                   acc_interval=[0.3, 1.0])

    # cifar10_lenet_dfs = get_train_val_test_compute_experiment_results(cifar10_lenet_exp)
    # cifar10_conv4_dfs = get_train_val_test_compute_experiment_results(cifar10_conv4_exp)
    # cifar10_conv6_dfs = get_train_val_test_compute_experiment_results(cifar10_conv6_exp)

    #
    # plot_bv_results(cifar10_lenet_exp, err_interval=[0.0, 0.8], title="LeNet Bias-Variance on CIFAR10")
    # plot_bv_results(cifar10_conv4_exp, err_interval=[0.0, 0.8], title="Conv4 Bias-Variance on CIFAR10")
    # plot_bv_results(cifar10_conv6_exp, err_interval=[0.0, 0.8], title="Conv6 Bias-Variance on CIFAR10")
    #
    # cifar10_lenet_dfs = get_train_val_test_compute_experiment_results(cifar10_lenet_exp)
    # cifar10_conv4_dfs = get_train_val_test_compute_experiment_results(cifar10_conv4_exp)
    # cifar10_conv6_dfs = get_train_val_test_compute_experiment_results(cifar10_conv6_exp)
    #
    # cifar10_dfs_dict = dict(lenet=cifar10_lenet_dfs, conv4=cifar10_conv4_dfs, conv6=cifar10_conv6_dfs)
    #
    # plot_train_val_test_compute_experiment_results(cifar10_dfs_dict, num_runs=exp_num_runs, title="CIFAR10", acc_interval=[0.3, 1.0])



