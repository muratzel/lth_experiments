import torch
from torch import Tensor
from torch.nn import Module

import os
import collections
from typing import List, Tuple, Any, OrderedDict

from LitModule import LitModuleArgs, LitModule


def _get_layers_with_weights(module: Module, return_dict: dict = None, name: str = "") -> None:

    if not module._modules:
        if module._parameters:
            return_dict[name + "_" + module.__class__.__name__.lower()] = (module, tuple(module._parameters.keys()))
        return

    for child_name, child_model in module._modules.items():
        if name:
            child_name = name + "_" + child_name
        _get_layers_with_weights(child_model, return_dict, child_name)


def _get_layers_with_masks(masks: dict, layers_with_weights: OrderedDict,
                           ignore_layer_names: List[str], ignore_layer_types: List[Module],
                           ignore_weight_names: List[str]) -> None:

    ignore_layer_names = set(ignore_layer_names)
    ignore_layer_types = tuple(ignore_layer_types)
    ignore_weight_names = set(ignore_weight_names)

    for layer_name, (layer, params) in layers_with_weights.items():

        layer_name_parts = [name_part for name_part in layer_name.split("_") if not name_part.isnumeric()]
        if any(name_part in ignore_layer_names for name_part in layer_name_parts) or \
                isinstance(layer, ignore_layer_types):
            continue

        layer_data = (layer, dict())
        for param in params:
            if param not in ignore_weight_names:
                layer_data[1][param] = torch.ones_like(layer._parameters[param])
        if layer_data[1]:
            masks[layer_name] = layer_data


def _gather_weights(layers_with_weights: OrderedDict) -> torch.Tensor:

    weights_list = []
    for layer, params in layers_with_weights.values():
        for param in params:
            weights_list.append(layer._parameters[param].flatten())
    return torch.concat(weights_list)


def _unflatten_masks(layers_with_weights: OrderedDict, masks: dict, flat_mask: torch.Tensor) -> None:

    cur_pos = 0
    for layer_name, (layer, params) in layers_with_weights.items():
        layer_masks = masks.get(layer_name, (None, dict()))[1]
        for param in params:
            end_pos = cur_pos + layer._parameters[param].numel()
            if param in layer_masks:
                layer_masks[param] = flat_mask[cur_pos:end_pos].reshape(layer_masks[param].shape)
            cur_pos = end_pos


def _top_k_strategy_mask(weight_scores: torch.Tensor, pct: float) -> torch.Tensor:

    top_k = int(pct * weight_scores.shape[0])
    top_k_indices = torch.topk(weight_scores, top_k, sorted=False).indices

    flat_mask = torch.zeros_like(weight_scores)
    flat_mask[top_k_indices] = 1.0

    return flat_mask


def _apply_masks(masks):

    for (layer, layer_masks) in masks.values():
        for param, mask in layer_masks.items():
            layer._parameters[param].data *= mask.to(layer._parameters[param].data.device)


def _replace_masks(cur_masks, new_masks):
    for layer_name, (_, layer_masks) in cur_masks.items():
        for param in layer_masks.keys():
            layer_masks[param] = new_masks[layer_name][1][param]


class MaskedModuleArgs(LitModuleArgs):

    use_mask: bool

    mask_strategy: str
    mask_strategy_kwargs: dict

    mask_ignore_layer_names: List[str]
    mask_ignore_layer_types: List[Module]
    mask_ignore_weight_names: List[str]

    random_reinit: bool

    init_weights_path: str
    masks_path: str

    def to_dict(self) -> dict:
        args_dict = super(MaskedModuleArgs, self).to_dict()
        args_dict.update(dict(mask_strategy=self.mask_strategy,
                              mask_strategy_kwargs=self.mask_strategy_kwargs,
                              mask_ignore_layer_names=self.mask_ignore_layer_names,
                              mask_ignore_layer_types=self.mask_ignore_layer_types,
                              mask_ignore_weight_names=self.mask_ignore_weight_names))
        return args_dict


class ComputeMaskModule(LitModule):

    def __init__(self, module_args: MaskedModuleArgs, sub_modules: OrderedDict[str, Tuple[Module, Any]]):

        super(ComputeMaskModule, self).__init__(module_args, sub_modules)

        self.layers_with_weights = collections.OrderedDict()    # layer_name -> (layer, param_names)
        _get_layers_with_weights(self, self.layers_with_weights)

        self.masks = {}     # layer_name -> (layer, dict(param_name -> mask))
        _get_layers_with_masks(self.masks, self.layers_with_weights,
                               module_args.mask_ignore_layer_names,
                               module_args.mask_ignore_layer_types,
                               module_args.mask_ignore_weight_names)

        self.init_weights_path = None
        self.masks_path = None

        self.mask_strategy = module_args.mask_strategy
        self.mask_strategy_kwargs = module_args.mask_strategy_kwargs
        self.mask_ignore_layer_names = module_args.mask_ignore_layer_names
        self.mask_ignore_layer_types = module_args.mask_ignore_layer_types
        self.mask_ignore_weight_names = module_args.mask_ignore_weight_names

    def forward(self, x) -> Tensor:

        for sub_module in self._modules.values():
            x = sub_module(x)
        return x

    def on_train_start(self) -> None:

        try:
            self.init_weights_path = self.trainer.checkpoint_callback.dirpath + "/init_weights"
            self.masks_path = self.trainer.checkpoint_callback.dirpath + "/masks"
            os.mkdir(self.trainer.checkpoint_callback.dirpath)
        except RuntimeError:
            self.init_weights_path = self.trainer.default_root_dir + "/init_weights"
            self.masks_path = self.trainer.default_root_dir + "/masks"

        torch.save(self.state_dict(), self.init_weights_path)

    def on_train_end(self) -> None:

        if self.mask_strategy == "abs_final":

            assert "pct" in self.mask_strategy_kwargs

            all_weights = _gather_weights(self.layers_with_weights)
            flat_mask = _top_k_strategy_mask(torch.abs(all_weights), self.mask_strategy_kwargs["pct"])

            self.load_state_dict(torch.load(self.init_weights_path))

        elif self.mask_strategy == "abs_change":

            assert "pct" in self.mask_strategy_kwargs

            cur_weights = _gather_weights(self.layers_with_weights)

            self.load_state_dict(torch.load(self.init_weights_path))
            init_weights = _gather_weights(self.layers_with_weights)

            flat_mask = _top_k_strategy_mask(torch.abs(cur_weights - init_weights), self.mask_strategy_kwargs["pct"])

        else:

            raise ValueError("Strategy has to be abs_final or abs_change")

        _unflatten_masks(self.layers_with_weights, self.masks, flat_mask)
        torch.save(self.masks, self.masks_path)


class MaskedModule(LitModule):

    def __init__(self, module_args: MaskedModuleArgs, sub_modules: OrderedDict[str, Tuple[Module, Any]]):

        super(MaskedModule, self).__init__(module_args, sub_modules)

        self.layers_with_weights = collections.OrderedDict()  # layer_name -> (layer, param_names)
        _get_layers_with_weights(self, self.layers_with_weights)

        self.masks = {}  # layer_name -> (layer, dict(param_name -> mask))
        _get_layers_with_masks(self.masks, self.layers_with_weights,
                               module_args.mask_ignore_layer_names,
                               module_args.mask_ignore_layer_types,
                               module_args.mask_ignore_weight_names)

        self.random_reinit = module_args.random_reinit
        self.init_weights_path = module_args.init_weights_path
        self.masks_path = module_args.masks_path

    def forward(self, x) -> Tensor:

        _apply_masks(self.masks)

        for sub_module in self._modules.values():
            x = sub_module(x)
        return x

    def on_train_start(self) -> None:

        if not self.random_reinit:
            self.load_state_dict(torch.load(self.init_weights_path))
        _replace_masks(self.masks, torch.load(self.masks_path))
