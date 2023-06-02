from torch import Tensor, concat
from torch.nn import Module
from torch.utils.data import DataLoader

from lightning import Trainer

from typing import Tuple, Optional, OrderedDict, Any, Callable

from Data import DataSet
from LitModule import LitModuleArgs, LitModule


class LitTrainerArgs:

    root_dir: str

    database_name: str
    batch_size: Optional[int]
    shuffle: Optional[bool]
    train_val_split: Optional[Tuple[int, int]]

    callbacks: Optional[Callable]
    val_check_interval: Optional[int]
    check_val_every_n_epoch: Optional[int]
    max_epochs: Optional[int]

    def to_dict(self) -> dict:
        return dict(database_name=self.database_name, batch_size=self.batch_size, train_val_split=self.train_val_split)


class LitTrainer:

    def __init__(self,
                 module_type: LitModule,
                 module_args: LitModuleArgs,
                 sub_modules: OrderedDict[str, Tuple[Module, Any]],
                 trainer_args: LitTrainerArgs):

        self.module = module_type(module_args, sub_modules)
        self.module.save_hyperparameters(trainer_args.to_dict())

        if trainer_args.database_name is None:
            self.dataset = None
        else:
            self.dataset = DataSet(name=trainer_args.database_name,
                                   batch_size=trainer_args.batch_size,
                                   shuffle=trainer_args.shuffle,
                                   train_val_split=trainer_args.train_val_split)

        self.trainer = Trainer(accelerator="auto", devices="auto", strategy="auto",
                               enable_progress_bar=True,
                               enable_checkpointing=True,
                               default_root_dir=trainer_args.root_dir,
                               callbacks=trainer_args.callbacks(),
                               val_check_interval=trainer_args.val_check_interval,
                               check_val_every_n_epoch=trainer_args.check_val_every_n_epoch,
                               max_epochs=trainer_args.max_epochs)

    def train(self, train_loader: Optional[DataLoader] = None, val_loader: Optional[DataLoader] = None) -> None:

        if train_loader is None and val_loader is None:
            self.dataset.get_data_loaders()
            train_loader, val_loader = self.dataset.train_loader, self.dataset.test_loader

        self.trainer.fit(self.module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def train_bootstrap(self) -> None:
        self.dataset.get_bootstrap_loaders()
        self.trainer.fit(self.module,
                         train_dataloaders=self.dataset.train_loader,
                         val_dataloaders=self.dataset.val_loader)

    def test(self, test_loader: Optional[DataLoader] = None) -> None:
        if test_loader is None:
            test_loader = self.dataset.test_loader
        self.trainer.test(self.module, dataloaders=test_loader, ckpt_path="best")

    def predict(self,
                predict_inputs_loader: Optional[DataLoader] = None,
                predict_labels_loader: Optional[DataLoader] = None) -> [Tensor, Tensor]:

        if predict_inputs_loader is None and predict_labels_loader is None:
            predict_inputs_loader = self.dataset.predict_inputs_loader
            predict_labels_loader = self.dataset.predict_labels_loader

        predictions = concat(self.trainer.predict(self.module, dataloaders=predict_inputs_loader,
                                                  return_predictions=True, ckpt_path="best"))
        labels = concat([label_batch[0] for label_batch in predict_labels_loader])
        return predictions, labels


