from torch.optim import Optimizer
from torch.nn import Module

from lightning import LightningModule
from torchmetrics import Accuracy

from typing import Any, Tuple, OrderedDict


class LitModuleArgs:

    root_dir: str
    loss: Any

    optimizer: Optimizer
    optimizer_kwargs: dict

    def to_dict(self) -> dict:
        return dict(optimizer=self.optimizer, optimizer_kwargs=self.optimizer_kwargs)


class LitModule(LightningModule):

    def __init__(self, module_args: LitModuleArgs, sub_modules: OrderedDict[str, Tuple[Module, Any]]):

        super(LitModule, self).__init__()

        assert sub_modules

        for name, (sub_module, sub_module_args) in sub_modules.items():
            self.add_module(name, sub_module(sub_module_args))
            self.save_hyperparameters({name + "_args": sub_module_args.to_dict()})

        self.save_hyperparameters(module_args.to_dict())

        self.loss = module_args.loss
        self.output_shape = sub_module_args.output_shape

        self.optimizer = module_args.optimizer
        self.optimizer_kwargs = module_args.optimizer_kwargs

    def training_step(self, batch, batch_idx):
        out = self(batch[0])
        loss = self.loss(out, batch[1])
        return loss

    def validation_step(self, batch, batch_idx):

        out = self(batch[0])
        labels = batch[1]

        loss = self.loss(out, labels)
        acc = Accuracy("multiclass", num_classes=self.output_shape).to(out.device)(out, labels)

        log_dict = {"val_loss": loss, "val_acc": acc}

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        out = self(batch[0])
        labels = batch[1]

        loss = self.loss(out, labels)
        acc = Accuracy("multiclass", num_classes=self.output_shape).to(out.device)(out, labels)

        log_dict = {"test_loss": loss, "test_acc": acc}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)