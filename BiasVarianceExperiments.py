import torch
from torch.nn.functional import cross_entropy, one_hot
from torch.nn import BatchNorm1d, BatchNorm2d

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import os
import shutil
import collections
import glob
from itertools import chain
from typing import List, OrderedDict

from Modules import ConvNet, LeNet_params, Conv4_params, Conv6_params
from MaskedModule import MaskedModuleArgs, ComputeMaskModule, MaskedModule
from LitTrainer import LitTrainerArgs, LitTrainer
from ExperimentArgs import ExperimentArgs


def _set_dropout(exp_sub_modules, val):
    for _, sub_module_args in exp_sub_modules.values():
        for params in chain(*(params for params in sub_module_args.__dict__.values()
                              if isinstance(params, list))):
            if isinstance(params, dict) and "dropout" in params:
                params["dropout"] = val


class BVExperiment:

    def __init__(self, experiment_args: ExperimentArgs, num_runs: int = 5):

        self.experiment_args = experiment_args
        self.num_runs = num_runs
        self.root_dir = experiment_args.experiment_name
        self.use_reg = experiment_args.use_reg

        reg_sufix = "" if self.use_reg else "_wo_reg"
        self.root_dir = self.root_dir + reg_sufix

        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

    def run_experiment(self) -> None:

        reg_sufix = "" if self.use_reg else "_wo_reg"

        for exp_name, exp_module_args, exp_sub_modules, exp_trainer_args in self.experiment_args:

            no_mask_dir = self.root_dir + "/" + exp_name + "_no_mask"
            mask_dir = self.root_dir + "/" + exp_name + "_mask" + reg_sufix
            mask_rand_dir = self.root_dir + "/" + exp_name + "_mask_rand" + reg_sufix
            bv_results_dir = self.root_dir + "/" + exp_name + "_bv_results" + reg_sufix

            bv_predictions_dir = bv_results_dir + "/predictions"
            bv_labels_dir = bv_results_dir + "/labels"

            if not os.path.exists(bv_results_dir):
                os.mkdir(bv_results_dir)
            if not os.path.exists(bv_predictions_dir):
                os.mkdir(bv_predictions_dir)
            if not os.path.exists(bv_labels_dir):
                os.mkdir(bv_labels_dir)

            num_runs = 0
            if os.path.exists(no_mask_dir + "/lightning_logs"):
                num_runs = len(glob.glob(no_mask_dir + "/lightning_logs/*"))

            while num_runs < self.num_runs:

                if exp_name in ["conv4", "conv6"]:
                    module_args.optimizer_kwargs = dict(lr=0.0005, weight_decay=0.001)
                else:
                    module_args.optimizer_kwargs = dict(lr=0.001, weight_decay=0.001)
                _set_dropout(exp_sub_modules, True)

                exp_trainer_args.root_dir = no_mask_dir

                comp_mask_trainer = LitTrainer(ComputeMaskModule, exp_module_args, exp_sub_modules, exp_trainer_args)
                comp_mask_trainer.train()
                comp_mask_trainer.test()

                if module_args.mask_strategy_kwargs["pct"] < 1.0:

                    if not self.use_reg:
                        module_args.optimizer = torch.optim.Adam
                        module_args.optimizer_kwargs = dict(lr=0.001, weight_decay=0.0)
                        _set_dropout(exp_sub_modules, False)

                    exp_trainer_args.root_dir = mask_dir

                    exp_module_args.random_reinit = False
                    exp_module_args.masks_path = comp_mask_trainer.module.masks_path
                    exp_module_args.init_weights_path = comp_mask_trainer.module.init_weights_path

                    masked_trainer = LitTrainer(MaskedModule, exp_module_args, exp_sub_modules, exp_trainer_args)
                    masked_trainer.train(comp_mask_trainer.dataset.train_loader, comp_mask_trainer.dataset.val_loader)
                    masked_trainer.test()

                    exp_predictions, exp_labels = masked_trainer.predict()

                    shutil.rmtree(mask_dir + "/lightning_logs/version_" + str(num_runs) + "/checkpoints")

                    exp_trainer_args.root_dir = mask_rand_dir

                    exp_module_args.random_reinit = True
                    exp_module_args.masks_path = comp_mask_trainer.module.masks_path
                    exp_module_args.init_weights_path = comp_mask_trainer.module.init_weights_path

                    rand_masked_trainer = LitTrainer(MaskedModule, exp_module_args, exp_sub_modules, exp_trainer_args)
                    rand_masked_trainer.train(comp_mask_trainer.dataset.train_loader, comp_mask_trainer.dataset.val_loader)
                    rand_masked_trainer.test()

                    shutil.rmtree(mask_rand_dir + "/lightning_logs/version_" + str(num_runs) + "/checkpoints")

                else:
                    exp_predictions, exp_labels = comp_mask_trainer.predict()

                torch.save(exp_predictions, bv_predictions_dir + "/" + str(num_runs))

                exp_labels = one_hot(exp_labels, num_classes=comp_mask_trainer.module.output_shape)
                torch.save(exp_labels, bv_labels_dir + "/" + str(num_runs))

                shutil.rmtree(no_mask_dir + "/lightning_logs/version_" + str(num_runs) + "/checkpoints")
                num_runs += 1


def run_bv_experiments_on_database(database_name: str, experiment_names: List[str],
                                   experiment_sub_modules: List[OrderedDict]) -> None:

    trainer_args.database_name = database_name

    for exp_name, exp_sub_modules in zip(experiment_names, experiment_sub_modules):

        bv_experiment_args = ExperimentArgs(experiment_name=exp_name,
                                            module_args=module_args,
                                            sub_modules=exp_sub_modules,
                                            trainer_args=trainer_args,
                                            experiment_config_names=bv_experiment_config_names,
                                            experiment_configs=bv_experiment_configs,
                                            use_reg=use_reg)
        bv_experiment = BVExperiment(experiment_args=bv_experiment_args, num_runs=num_runs)
        bv_experiment.run_experiment()


# ##################### DEFINE SUBMODULES #############################

mnist_lenet_args = LeNet_params(input_shape=(1, 28, 28), output_shape=10)
mnist_lenet_sub_modules = collections.OrderedDict(dict(lenet=(ConvNet, mnist_lenet_args)))

cifar_lenet_args = LeNet_params(input_shape=(3, 32, 32), output_shape=10)
cifar_lenet_sub_modules = collections.OrderedDict(dict(lenet=(ConvNet, cifar_lenet_args)))

cifar_conv4_args = Conv4_params(input_shape=(3, 32, 32), output_shape=10)
cifar_conv4_sub_modules = collections.OrderedDict(dict(cifar_conv4=(ConvNet, cifar_conv4_args)))

cifar_conv6_args = Conv6_params(input_shape=(3, 32, 32), output_shape=10)
cifar_conv6_sub_modules = collections.OrderedDict(dict(cifar_conv6=(ConvNet, cifar_conv6_args)))

# ##################### DEFINE MODULE #############################

module_args = MaskedModuleArgs()

module_args.loss = cross_entropy
module_args.optimizer = torch.optim.AdamW
module_args.optimizer_kwargs = dict(lr=0.001, weight_decay=0.001)

module_args.mask_strategy = "abs_final"
module_args.mask_strategy_kwargs = None
module_args.mask_ignore_layer_names = []
module_args.mask_ignore_layer_types = [BatchNorm1d, BatchNorm2d]
module_args.mask_ignore_weight_names = []

# ##################### DEFINE TRAINING STUFF #############################

trainer_args = LitTrainerArgs()

trainer_args.root_dir = None
trainer_args.database_name = "mnist"
trainer_args.batch_size = 64
trainer_args.shuffle = True
trainer_args.train_val_split = (0.75, 0.25)
trainer_args.val_check_interval = 500
trainer_args.check_val_every_n_epoch = None
trainer_args.max_epochs = 50

trainer_args.callbacks = lambda: [ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                                                  filename="best_weights-{epoch:02d}"),
                                  EarlyStopping(monitor="val_loss", mode="min", patience=3,
                                                check_on_train_epoch_end=False)]

# ##################### EXPERIMENTS #############################

num_runs = 5
pcts = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.032, 0.016, 0.01]
use_reg = False
bv_experiment_config_names = ["pct"+str(pct) for pct in pcts]
bv_experiment_configs = dict(mask_strategy_kwargs=[dict(pct=pct) for pct in pcts])

run_bv_experiments_on_database(database_name="mnist",
                               experiment_names=["mnist_lenet2"],
                               experiment_sub_modules=[mnist_lenet_sub_modules])

run_bv_experiments_on_database(database_name="cifar10",
                               experiment_names=["cifar10_lenettest"],
                               experiment_sub_modules=[cifar_lenet_sub_modules])

