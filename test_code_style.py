#!/usr/bin/env python3
"""
Copyright 2021 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import os
import subprocess
import sys


def recursively_check(directory):
    """
    Recursively check a directory

    Parameters
    ----------
    directory: str
        The directory to be analyzed
    """
    logging.info('Analyzing %s', directory)
    to_be_skipped = ['venv']
    linters = ['pylint', 'mypy', 'flake8']
    for element in os.listdir(directory):
        if element in to_be_skipped:
            continue
        if os.path.isdir(os.path.join(directory, element)):
            if element.startswith('output'):
                continue
            recursively_check(os.path.join(directory, element))
        else:
            if not element.endswith('py'):
                continue
            for linter in linters:
                command = linter + " " + os.path.join(directory, element)
                logging.info('%s', command)
                return_value = subprocess.call(command, shell=True)
                if return_value == 0:
                    logging.info('SUCCESS')
                else:
                    logging.info('FAILURE')
                    sys.exit(return_value)


def main():
    """
    Script to apply linting to python code
    """
    # The absolute path of the current script
    abs_script = os.path.abspath(sys.argv[0])

    # The root directory of the script
    abs_root = os.path.dirname(abs_script)

    logging.basicConfig(level=logging.INFO)
    recursively_check(abs_root)


if __name__ == '__main__':
    main()
