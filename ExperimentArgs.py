from torch.nn import Module

from typing import Optional, Dict, List, Any, Iterator, OrderedDict, Tuple, Union

from LitModule import LitModuleArgs
from LitTrainer import LitTrainerArgs


class ExperimentArgs:

    def __init__(self,
                 experiment_name: str,
                 module_args: LitModuleArgs,
                 sub_modules: OrderedDict[str, Tuple[Module, Any]],
                 trainer_args: LitTrainerArgs,
                 experiment_config_names: Optional[Union[str, List[str]]] = None,
                 experiment_configs: Optional[Dict[Union[str, Tuple[str, str]], List[Any]]] = None,
                 use_reg: Optional[bool] = False):

        self.experiment_name = experiment_name
        self.experiment_config_names = experiment_config_names
        self.module_args = module_args
        self.sub_modules = sub_modules
        self.trainer_args = trainer_args
        self.experiment_configs = experiment_configs
        self.use_reg = use_reg

    def __iter__(self) -> Iterator[Tuple[str, LitModuleArgs, OrderedDict, LitTrainerArgs]]:

        if self.experiment_configs is None:
            yield self.experiment_name, self.module_args, self.sub_modules, self.trainer_args
        else:

            num_experiments = min(len(param_configs) for param_configs in self.experiment_configs.values())

            for i in range(num_experiments):

                for param, configs in self.experiment_configs.items():

                    if isinstance(param, tuple):
                        submodule_name, param = param
                        if submodule_name in self.sub_modules:
                            setattr(self.sub_modules[submodule_name][1], param, configs[i])
                        else:
                            raise KeyError(param + " doesn't exist")
                    elif hasattr(self.module_args, param):
                        setattr(self.module_args, param, configs[i])
                    elif hasattr(self.trainer_args, param):
                        setattr(self.trainer_args, param, configs[i])
                    else:
                        raise KeyError(param + " doesn't exist")

                yield self.experiment_name + "_" + self.experiment_config_names[i], \
                    self.module_args, self.sub_modules, self.trainer_args
