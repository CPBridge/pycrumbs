# `pycrumbs`

This is a Python package to create "record files" of whenever a function is
executed. This file is placed alongside the output files of your function and
allows you to "follow the breadcrumbs" to figure out how exactly a set of files
were created. This includes information on the parameters of the functions, the
platform and environment it was run in, the versions of all libraries
installed, and source control information about the current state of the
project (using git).

`pycrumbs` was developed for reproducible machine learning pipelines, but can
be used for any application where Python code creates output files that may
need to be reproduced at a later stage.

### Status

`pycrumbs` is currently in an early stage for a trial period. Ultimately, I
hope to list the module in PyPI when the API is more stable. Until that time,
there may be API changes at any time and for any reason.

### Installation

You can install the package directly from this repo using pip:

```
pip install git+https://github.com/CPBridge/pycrumbs
```

I recommend specifying a particular git hash to ensure you can re-install the
same version later (`9aa7825e` here is an example git hash, you should take
the latest from above):

```
pip install git+https://github.com/CPBridge/pycrumbs@9aa7825e
```

### Use

The sole purpose of this library is to create "record" files for a Python
functions that execute. The intended use of this is to "track" calls to key
functions (such as machine learning training or preprocessing routines) to
ensure that they can be reproduced later. The type of information captured
includes:

- Information about the function that was called. Its name, source file,
  and arguments it was called with at runtime.
- Environment information. Username that ran the code, the
  node and operating system on which it was run, GPUs available on the
  machine. SLURM specific information if run within a SLURM environment.
- Timing information (start and end times and duration).
- Command line arguments used to invoke the Python interpreter.
- Version control information about the function being executed, including
  the git hash and current active branch. Note that this requires that the
  function was executed from a source file within a git repository. If you
  want to install your package code with pip, use the
  `pip install --editable` option to allow for tracking with this
  decorator.
- A list of all Python packages currently installed, with their versions.
- Seeding information for random number generators.

You can very easily add a job record to a function using the `tracked`
decorator in the `pycrumbs` module. It is assumed
that a tracked function will create output files within a given output
directory, and that the job record file (a JSON file) should be placed in
the same directory. `tracked` gives you various options on how to
specify where the output location is, including intercepting runtime
parameters of the decorated function:

Specify a literal output location (always the same).

```python
from pathlib import Path
from pycrumbs import tracked


@tracked(literal_directory=Path('/home/user/proj/'))
def my_train_fun():
    # Do something...
    pass

# Record will be placed at /home/user/proj/my_train_fun_record.json
my_train_fun()
```

Specify an output location by intercepting the runtime value of a parameter
of the decorated function.

```python
from pathlib import Path
from pycrumbs import tracked


@tracked(dirctory_parameter='model_output_dir'))
def my_train_fun(model_output_dir: Path):
    # Do something...
    pass

# Record will be placed at
# /home/user/proj/my_model/my_train_fun_record.json
my_train_fun(model_output_dir=Path('/home/user/proj/my_model'))
```

Specify an output location that is determined dynamically by intercepting
the runtime value of a parameter of the decorated function to use as the
sub-directory of a literal location. This is useful when only the final
part of the path changes between different times the function is run.

```python
from pathlib import Path
from pycrumbs import tracked


@tracked(
    literal_directory=Path('/home/user/proj/'),
    subdirectory_name_parameter='model_name'
)
def my_train_fun(model_name: str):
    # Do something...
    pass

# Record will be placed at
# /home/user/proj/my_model/my_train_fun_record.json
my_train_fun(model_name='my_model')
```

You may also have the decorator dynamically add a timestamp
(`include_timestamp`) or a UUID (`include_uuid`) to the output directory to
ensure that each run results in a unique output directory. If you do this
in combination with `output_dir_parameter` or `subdirectory_name_parameter`, the
value of the relevant parameter will be updated to reflect the addition
of the UUID/timestamp.

```python
from pathlib import Path
from pycrumbs import tracked

@tracked(
    literal_directory=Path('/home/user/proj/'),
    subdirectory_name_parameter='model_name',
    include_uuid=True,
)
def my_train_fun(model_name: str):
    print(model_name)
    # Do something...
    pass

# Record will be placed at, e.g.
# /home/user/proj/my_model_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2/my_train_fun_record.json
my_train_fun(model_name='my_model')
# prints my_model_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2
```

Alternatively, you may specify an alternative parameter of the wrapped
function into which the full updated output directory path will be placed.
Use the `directory_injection_parameter` to specify this. This is required when
you append a timestamp or a UUID without specifying an `output_dir_parameter`
or `subdirectory_name_parameter`, otherwise there is no way for the wrapped
function to access the updated output directory. However, you may use
this at any time for your convenience.

```python
from pathlib import Path
from pycrumbs import tracked

@tracked(
    literal_directory=Path('/home/user/proj/'),
    include_uuid=True,
    directory_injection_parameter='model_directory'
)
def my_train_fun(
    model_directory: Path | None
):
    print(model_directory)
    # Do something...
    pass

# Record will be placed at, e.g.
# /home/user/proj_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2/my_train_fun_record.json
my_train_fun()
# prints /home/user/proj_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2
```

#### Seeds

In addition to tracking your function call, `pycrumbs` also handles seeding
random number generators for you and storing the seed in the record file
so that you can reproduce the run later, if needed. `pycrumbs` knows how to
seed random number generators in the following libraries and will do so if
they are installed:

- The Python standard library `random` module.
- Numpy
- Tensorflow
- Pytorch

Additionally you may specify a seed parameter by intercepting the runtime value
of a parameter of the decorated function. This is always recommended as it
allows re-running the job with the same seed without having to change the code.

```python
from pathlib import Path
from pycrumbs import tracked


@tracked(
    literal_directory=Path('/home/user/proj/'),
    subdirectory_name_parameter='model_name',
    seed_parameter='seed'
)
def my_train_fun(model_name: str, seed: int | None = None):
    # Do something...
    print(seed)

# Seed will be injected at runtime by the decorator mechanism and stored in the
# record file, you don't have to do anything else
my_train_fun(model_name='my_model')
# prints e.g. 272428

# But you can manually specify the seed later to reproduce, without
# having to alter the function signature
my_train_fun(model_name='my_model', seed=272428)
# prints 272428
```

#### Combining with Other Decorators

If you use this decorator in combination with decorators such as those from the
`click` module (and most other common decorators), you should place this
decorator last. This is because the `click` decorators alter the function
signature, which will break the operation of `tracked`. The last decorator is
applied first, meaning that it will operate on the function with its original
signature as intended.

```python
from pathlib import Path
import click
from pycrumbs import tracked


@click.command()
@click.argument('model_name')
@click.option('--seed', '-s', type=int, help='Random seed.')
@tracked(
    literal_directory=Path('/home/user/proj/'),
    subdirectory_name_parameter='model_name',
    seed_parameter='seed'
)
def my_train_fun(model_name: str, seed: int | None = None):
    # Do something...
    print(seed)
```

### Record Files

Here is an example of a job record file created by running a simple function
like the ones in the example above:

```json
{
    "uuid": "51af8c51-0de5-48f5-a7da-9ec9688d56b6",
    "timing": {
        "start_time": "2023-01-16 22:38:54.484040",
        "end_time": "2023-01-16 22:38:54.534795",
        "run_time": "0:00:00.050755"
    },
    "environment": {
        "argv": [
            "thing.py"
        ],
        "orig_argv": [
            "/Users/chris/.pyenv/versions/pycrumbs/bin/python",
            "thing.py"
        ],
        "platform": "darwin",
        "platform_info": "macOS-12.3.1-arm64-arm-64bit",
        "python_version": "3.10.3 (main, Sep 26 2022, 17:17:59) [Clang 13.1.6 (clang-1316.0.21.2.5)]",
        "python_implementation": "cpython",
        "python_executable": "/Users/chris/.pyenv/versions/pycrumbs/bin/python",
        "cwd": "/Users/chris/Developer/project",
        "hostname": "hostname.example.com",
        "python_path": [
            "/Users/chris/Developer/project",
            "/Users/chris/.pyenv/versions/project/lib/python3.10/site-packages/git/ext/gitdb",
            "/Users/chris/.pyenv/versions/3.10.3/lib/python310.zip",
            "/Users/chris/.pyenv/versions/3.10.3/lib/python3.10",
            "/Users/chris/.pyenv/versions/3.10.3/lib/python3.10/lib-dynload",
            "/Users/chris/.pyenv/versions/project/lib/python3.10/site-packages",
            "/Users/chris/Developer/project/src"
        ],
        "cpu_count": 10,
        "user": "cpb28",
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": null,
            "VIRTUAL_ENV": "/Users/chris/.pyenv/versions/3.10.3/envs/pycrumbs",
            "PYENV_VIRTUAL_ENV": "/Users/chris/.pyenv/versions/3.10.3/envs/pycrumbs",
            "PYTHONPATH": null
        }
    },
    "package_inventory": {
        "setuptools": "65.6.3",
        "types-setuptools": "57.4.0",
        "attrs": "22.1.0",
        "pip": "22.0.4",
        "packaging": "22.0",
        "ipython": "7.34.0",
        "pytest": "7.2.0",
        "traitlets": "5.8.0",
        "decorator": "5.1.1",
        "smmap": "5.0.0",
        "pexpect": "4.8.0",
        "typing-extensions": "4.4.0",
        "gitdb": "4.0.10",
        "flake8": "3.7.7",
        "gitpython": "3.1.29",
        "prompt-toolkit": "3.0.36",
        "pydocstyle": "3.0.0",
        "pygments": "2.13.0",
        "pycodestyle": "2.5.0",
        "snowballstemmer": "2.2.0",
        "pyflakes": "2.1.1",
        "tomli": "2.0.1",
        "six": "1.16.0",
        "py": "1.9.0",
        "flake8-docstrings": "1.3.0",
        "iniconfig": "1.1.1",
        "exceptiongroup": "1.0.4",
        "flake8-polyfill": "1.0.2",
        "pluggy": "1.0.0",
        "mypy": "0.910",
        "jedi": "0.18.2",
        "toml": "0.10.2",
        "parso": "0.8.3",
        "pep8-naming": "0.8.2",
        "pickleshare": "0.7.5",
        "ptyprocess": "0.7.0",
        "mccabe": "0.6.1",
        "mypy-extensions": "0.4.3",
        "entrypoints": "0.3",
        "wcwidth": "0.2.5",
        "backcall": "0.2.0",
        "matplotlib-inline": "0.1.6",
        "appnope": "0.1.3",
        "pycrumbs": "0.1.0"
    },
    "called_function": {
        "name": "my_train_fun",
        "module": "__main__",
        "source_file": "/Users/chris/Developer/project/train.py",
        "parameters": {
            "model_name": "my_model",
            "seed": null
        },
        "altered_parameters": {
            "model_name": "my_model",
            "seed": 106543
        }
    },
    "tracked_module": {
        "module_path": "/Users/chris/Developer/project/train.py",
        "name": "__main__",
        "git_active_branch": "master",
        "git_commit_hash": "2a38ea37ee41466612f312d082445528df9f8d9d",
        "git_is_dirty": true,
        "git_remotes": {
            "origin": "ssh://git@github.com:/example/project.git"
        },
        "git_working_dir": "/Users/chris/Developer/project"
    },
    "seed": 106543
}
```
