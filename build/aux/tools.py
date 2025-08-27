import argparse
import pathlib
import subprocess
import typing

import constants

def exec(
    *args: pathlib.Path | str, 
    **kwargs: dict[str, any]
) -> subprocess.CompletedProcess: 
    cmd = ' '.join([*map(str, args)])
    return subprocess.run(cmd, shell = True, check = True, **kwargs)

def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = constants.DESCRIPTION)
    parser.add_argument(
        '-p', 
        '--prefix', 
        help = 'ReliQ installation path',
        type = str,
        default = constants.CWD / 'deps'
    )
    flags = ['--spack', '--nim', '--upcxx', '--kokkos']
    helpers = [
        'Path to Spack',
        'Path to Nim',
        'Path to UPC++',
        'Path to Kokkos'
    ]
    defaults = [
        constants.DEFAULT_STANDIN_PATH,
        constants.DEFAULT_STANDIN_PATH,
        constants.DEFAULT_STANDIN_PATH,
        constants.DEFAULT_STANDIN_PATH,
    ]
    for (flag, help, default) in zip(flags, helpers, defaults):
        parser.add_argument(flag, help = help, type = str, default = default)
    return parser.parse_args()