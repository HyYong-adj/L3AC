import argparse
from functools import reduce
from pathlib import Path
from typing import Any


class Parser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        for key, value in args.__dict__.items():
            if isinstance(value, str) and '~' in value:
                args.__dict__[key] = value.replace('~', str(Path.home()))
        return args


class AutoSigner:
    def __call__(self, func_obj):
        import inspect
        args_name = inspect.getfullargspec(func_obj).args

        if args_name[0] == "self":
            def wrapper(obj, *args: dict[str: Any], **kwargs):
                return func_obj(obj, **self.get_kwargs(args_name[1:], kwargs, *args))
        else:
            def wrapper(*args: dict[str: Any], **kwargs):
                return func_obj(**self.get_kwargs(args_name, kwargs, *args))

        return wrapper

    def get_kwargs(self, args_name: list[str], *args_dict: dict[str: Any]) -> dict[str, Any]:
        # Support both dict-style args (merging them) and positional tensor args.
        dicts = [d for d in args_dict if isinstance(d, dict)]
        others = [o for o in args_dict if not isinstance(o, dict)]

        merged: dict[str, Any] = {}
        if dicts:
            merged = reduce(lambda a, b: a | b, dicts)

        # Fill in positional arguments for parameters not present in merged
        kwargs: dict[str, Any] = dict(merged)
        pos_idx = 0
        for name in args_name:
            if name in kwargs:
                continue
            if pos_idx < len(others):
                kwargs[name] = others[pos_idx]
                pos_idx += 1
            else:
                kwargs[name] = None

        # Apply per-arg formatting if needed
        kwargs = {name: self.format_args(name, kwargs.get(name)) for name in args_name}
        return kwargs

    @staticmethod
    def format_args(arg_name, arg_value):
        return arg_value
