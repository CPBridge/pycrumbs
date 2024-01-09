"""Utilities to create and store records of jobs."""
from copy import deepcopy
import datetime
import functools
from getpass import getuser
import inspect
import json
import multiprocessing as mp
from os import environ, getcwd
from pathlib import Path
import importlib
import pkg_resources
import platform
import random
import sys
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Union, Sequence, cast, List
from uuid import uuid4

import git


_TIMESTAMP_FMT = '%Y_%m_%d_%H_%M_%S'
DOCKER_BUILD_HASH_LOCATION = Path('/etc/docker-build-git-hash')


def _format_json(obj: Any, char_limit: Optional[int] = None) -> Any:
    """Normalize arbitrary objects to ensure they are JSON serializable.

    Parameters
    ----------
    obj: Any
        Any Python object.
    char_limit: Optional[int]
        Character limit above which the object will be replaced with a
        placeholder string.

    Returns
    -------
    Any:
        In the case where obj is directly serializable, it will be returned
        unchanged. Otherwise a string representation of the object will be
        returned.

    """
    try:
        string = json.dumps(obj)
        serializable = True
    except (TypeError, OverflowError):
        string = repr(obj)
        serializable = False

    if char_limit is not None and len(string) > char_limit:
        string = '<suppressed due to excessive length>'
        return string
    else:
        if serializable:
            return obj
        else:
            return string


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment in a dictionary.

    The following information is included:
    - hostname on which the job was executed
    - information about the SLURM job, if this process is running inside SLURM
    - username that submitted the job
    - date and time the job was run
    - list of GPUs that were available to the job
    - list of command line arguments that began the process

    Returns
    -------
    dict
        Dictionary containing the relevant information.

    """
    info: Dict[str, Any] = {}

    # Store various pieces of system/platform information
    info['argv'] = sys.argv
    try:
        # Only available in Python 3.10+
        info['orig_argv'] = sys.orig_argv  # type: ignore
    except AttributeError:
        pass
    info['platform'] = sys.platform
    info['platform_info'] = platform.platform()
    info['python_version'] = sys.version
    info['python_implementation'] = sys.implementation.name
    info['python_executable'] = sys.executable
    info['cwd'] = getcwd()
    info['hostname'] = platform.node()
    info['python_path'] = sys.path
    info['cpu_count'] = mp.cpu_count()
    try:
        info['user'] = getuser()
    except Exception:
        info['user'] = ''

    # This file is created during container the build process for repos
    # created with the cookiecutter-qtim tool and captures the git commit hash
    # of the repo containing the docker file upon the building of the Docker
    # image
    if DOCKER_BUILD_HASH_LOCATION.exists():
        with DOCKER_BUILD_HASH_LOCATION.open() as f:
            docker_build_hash: Optional[str] = f.read().splitlines()[0]
        info['git_hash_at_docker_build'] = docker_build_hash

    return info


def get_environment_vars(
    extra_environment_variables: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Get values for some environment variables.

    There is a pre-defined list, and the user may additionally specify
    further values.

    Parameters
    ----------
    extra_environment_variables: Optional[Sequence[str]]
        Names of extra environment variables to include in the job record.

    Returns
    -------
    dict
        Dictionary containing a mapping of variable name to value.

    """
    # A list of environment variables to store
    vars_dict = {}
    var_list = [
        'CUDA_VISIBLE_DEVICES',
        'VIRTUAL_ENV',
        'PYENV_VIRTUAL_ENV',
        'PYTHONPATH',
    ]
    if 'SLURM_JOB_ID' in environ:
        var_list += [v for v in environ if v.startswith('SLURM')]
    if extra_environment_variables is not None:
        var_list += extra_environment_variables
    for var in var_list:
        vars_dict[var] = environ.get(var)

    return vars_dict


try:
    import tensorflow as tf  # noqa: F401
    _have_tensorflow = True
except ImportError:
    _have_tensorflow = False

try:
    import torch  # noqa: F401
    _have_torch = True
except ImportError:
    _have_torch = False

try:
    import numpy as np  # noqa: F401
    _have_numpy = True
except ImportError:
    _have_numpy = False


def seed_tasks(seed: Optional[int] = None) -> int:
    """Set random seeds for various libraries.

    Currently the following libraries are implemented:
        - The standard library's 'random' module
        - Numpy (if it is installed)
        - Tensorflow (if it is installed)
        - Pytorch (if it is installed)

    Parameters:
    -----------
    seed: Optional[int]
        Value to use as a random seed. Must be a positive integer.
        If no value is provided, a random seed is generated and returned.

    Returns
    -------
    int:
        The random seed. If a value was provided, it will be returned
        unchanged. If no value was provided, a random seed will be generated
        and returned.

    """
    if seed is None:
        seed = random.randint(0, 1000000)

    # Standard library random module
    random.seed(seed)

    # Set seeds
    # Numpy
    if _have_numpy:
        import numpy as np  # noqa: F811
        np.random.seed(seed)

    # Tensorflow
    if _have_tensorflow:
        import tensorflow as tf  # noqa: F811
        tf.random.set_seed(seed)

    # Pytorch
    if _have_torch:
        import torch  # noqa: F811
        torch.manual_seed(seed)

    return seed


def get_git_info(
    module: Union[str, ModuleType],
    allow_dirty: bool = False
) -> Dict[str, str]:
    """Get a record containing information about a module's git repository.

    Parameters
    ----------
    module: Union[str, ModuleType]
        Module, in the form of a name or runtime module object.
    allow_dirty: bool
        If False, an exception is raised if the module is dirty.

    Returns
    -------
    Dict[str, str]:
        Dictionary containing the following keys: git_commit_hash,
        git_active_branch, git_is_dirty, git_remotes, git_working_dir

    """
    # Get the current git commit, branch and state
    no_file_str = (
        "You appear to be trying to use git tracking on a function or method "
        "that is not defined in a file, which is not possible. This may be "
        "because the function is defined in an interactive Python shell."
    )
    if isinstance(module, str):
        mod = importlib.import_module(module)
        module_name = mod.__name__
        if not hasattr(mod, '__file__'):
            raise RuntimeError(no_file_str)
        module_path = mod.__file__
    else:
        module_name = module.__name__
        if not hasattr(module, '__file__'):
            raise RuntimeError(no_file_str)
        module_path = module.__file__

    try:
        repo = git.Repo(module_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        raise RuntimeError(
            f"Module '{module_name}' at path '{module_path}' is not "
            "within a git repository"
        )

    if not allow_dirty and repo.is_dirty():
        raise RuntimeError(
            f"Repository for module '{module_name}' is dirty (has "
            "uncommitted changes)."
        )

    record: Dict[str, Any] = {}
    record['module_path'] = module_path
    record['name'] = module_name
    try:
        record['git_active_branch'] = str(repo.active_branch)
    except TypeError:
        # We are in detached state
        record['git_active_branch'] = 'detached'
    record['git_commit_hash'] = str(repo.head.commit)
    record['git_is_dirty'] = repo.is_dirty()
    record['git_remotes'] = {item.name: item.url for item in repo.remotes}
    record['git_working_dir'] = repo.working_dir

    return record


def get_installed_packages() -> Dict[str, str]:
    """Get a list of all currently installed packages and their versions.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping package name to package version for all
        currently installed modules that have version information.

    """
    return {
        p.key: p.version for p in pkg_resources.working_set
    }


def write_record(
    record_path: Path,
    record: Dict[str, Any]
) -> None:
    """Save a job record file into an existing output directory.

    Parameters
    ----------
    record_path: Path
        Path to the output file. Directory must exist.
    record: Optional[Dict[str: Any]]
        A job record, as a dictionary.

    """
    # Save the record to file
    if not record_path.name.endswith('.json'):
        record_path = record_path.with_name(record_path.stem + '.json')
    with record_path.open('w') as jf:
        json.dump(record, jf, indent=4)


def tracked(
    *,
    literal_directory: Optional[Union[Path, str]] = None,
    directory_parameter: Optional[str] = None,
    subdirectory_name_parameter: Optional[str] = None,
    directory_injection_parameter: Optional[str] = None,
    include_timestamp: bool = False,
    include_uuid: bool = False,
    record_filename: Optional[str] = None,
    extra_modules: Optional[Sequence[Union[str, ModuleType]]] = None,
    extra_environment_variables: Optional[Sequence[str]] = None,
    seed_parameter: str = None,
    disable_git_tracking: bool = False,
    allow_dirty_repo: bool = True,
    include_package_inventory: bool = True,
    create_parents: bool = False,
    require_empty_directory: bool = False,
    chain_records: bool = False
) -> Callable:
    """Store information about a function call to disc.

    Use this as a decorator to a function that should be tracked (e.g. a
    training, preprocessing or inference routine). Various information
    about the function call will then be written to disc in the form of
    a job record in JSON format. The information captured includes:

    - Details of the function that was called. Its name, source file,
      and runtime arguments.
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
    - A list of all packages currently loaded, with their versions.

    Parameters
    ----------
    literal_directory: Optional[Union[Path, str]]
        Directory to store the record file in, provided as a literal value.
        Generall this means that the output directory will be the same every
        time the function runs, unless the value is altered before/as the
        pycrumbs library is imported.
    directory_parameter: Optional[Path]
        Directory to store the record file in, provided as the name of a
        function parameter whose value should be used as the output directory.
        In this way, the output directory depends upon the runtime values with
        which the function is called. The decorated function must have a
        parameter matching the name provided here.
    subdirectory_name_parameter: Optional[Path]
        Name of a subdirectory of the output directory (specified either
        directly by literal_directory or indirectly via directory_parameter) in
        which to place the record, provided as the name of a function parameter
        whose value should be used as the name of the output sub-directory.
        The decorated function must have a parameter matching the name provided
        here. If None, no sub-directory is used, and the record file is placed
        directly in the output directory.
    record_filename: Optional[str]
        Name of the record file within the output directory. The .json
        extension is optional and implied. If None, the record name will be
        "<function>_record.json" where <function> is the name of the decorated
        function.
    extra_modules: Optional[Sequence[Union[str, ModuleType]]]
        Additional module (provided as either a name or module object) for
        which tracking information should be included in the job record. Any
        package to be tracked in this way must be installed from a directory
        that exists within a git repository.
    extra_environment_variables: Optional[Sequence[str]]
        Names of extra environment variables to include in the job record.
    seed_parameter: str
        Name of the parameter of the decorated function to use as a random
        seed. If this parameter is given an integer value at runtime, that
        value will be used to set the random seed for several key libraries
        (see documentation of seed_tasks for full list) without any further
        action required by the user code. If instead the runtime value of
        the named parameter is None, a seed will be randomly generated and
        stored, and then injected into the user's function as if the function
        had been called with this seed value. The intended use of this
        mechanism is that during normal use, the decorator is allowed to
        generate its own seed, and a specific value for seed is only provided
        when recreating previous jobs.
    disable_git_tracking: bool
        Do not include version control information for the decorated function.
        This allows the use of this decorator when the decorated function
        is not under version control (not recommended). Any modules specified
        via extra_modules will still be tracked.
    allow_dirty_repo: bool
        If False, a runtime error will be raised if the decorated function is
        called when there are uncommitted changes within the repository to
        which it belongs, or there are uncommitted changes within any of the
        packages specified by 'extra_modules'. This 'strict' setting
        allows you to be sure that the code that is run can be re-run later.
    include_package_inventory: bool
        Include the inventory of all loaded modules with version information
        (this can get quite long).
    create_parents: bool
        Create the parents of the output directory if they do not exist.
    require_empty_directory: bool
        If True, an error will be raised if the output directory is not empty.
    include_timestamp: bool
        If True, a timestamp will be appended to the output directory name.
        Incompatible with include_uuid. If either directory_parameter or
        subdirectory_name_parameter are given (and
        directory_injection_parameter) is not given, the value of the resulting
        output directory name (or subdirectory name as appropriate) will
        replace the original value in the call to the wrapped function so that
        the wrapped function can access the updated output directory. If you
        wish to disable this, specify the directory_injection_parameter to
        specify the name of another parameter where you would like the updated
        directory name to be injected.
    include_uuid: bool
        If True, the UUID within the record will be appended to the output
        directory name. This should guarantee uniqueness of the output
        directory name if the same job is run with the same parameters multiple
        times. Incompatible with include_timestamp. See include_timestamp for
        an explanation of how the resulting output directory is supplied to the
        wrapped function.
    directory_injection_parameter: Optional[Path]
        Inject the final output directory into the decorated function in this
        parameter as a pathlib.Path object. The decorated function must have a
        parameter matching the name provided here. This is required when the
        include_timestamp or include_uuid options are selected and both
        directory_parameter and subdirectory_name_parameter are not specified
        so that the called decorated function is aware of the location of the
        output directory. It may be used in other situations for convenience.
    chain_records: bool
        If True, a pre-existing record file will have a new record appended to
        it within the same file. If False, a pre-existing record file will be 
        overwritten.

    Examples
    --------
    Specify a literal output location (always the same every time the function
    is run).

    >>> from pathlib import Path
    >>> from pycrumbs import tracked
    >>>
    >>> @tracked(literal_directory=Path('/home/user/proj/'))
    >>> def my_train_fun():
    >>>     # Do something...
    >>>     pass
    >>>
    >>> # Record will be placed at /home/user/proj/my_train_fun_record.json
    >>> my_train_fun()

    Specify an output location that is determined dynamically by intercepting
    the runtime value of a parameter of the decorated function.

    >>> from pathlib import Path
    >>> from pycrumbs import tracked
    >>>
    >>> @tracked(directory_parameter='model_output_dir'))
    >>> def my_train_fun(model_output_dir: Path):
    >>>     # Do something...
    >>>     pass
    >>>
    >>> # Record will be placed at
    >>> # /home/user/proj/my_model/my_train_fun_record.json
    >>> my_train_fun(model_output_dir=Path('/home/user/proj/my_model'))

    Specify an output location that is determined dynamically by intercepting
    the runtime value of a parameter of the decorated function to use as the
    sub-directory of a literal location. This is useful when only the final
    part of the path changes between different times the function is run.

    >>> from pathlib import Path
    >>> from pycrumbs import tracked
    >>>
    >>> @tracked(
    ...     literal_directory=Path('/home/user/proj/'),
    ...     subdirectory_name_parameter='model_name'
    ... )
    >>> def my_train_fun(model_name: str):
    >>>     # Do something...
    >>>     pass
    >>>
    >>> # Record will be placed at
    >>> # /home/user/proj/my_model/my_train_fun_record.json
    >>> my_train_fun(model_name='my_model')

    You may also have the decorator dynamically add a timestamp
    (include_timestamp) or a UUID (include_uuid) to the output directory to
    ensure that each time the function is called, a unique output directory is
    created. If you do this in combination with directory_parameter or
    subdirectory_name_parameter, the value of the relevant parameter will be
    updated to reflect the addition of the UUID/timestamp.

    >>> from pathlib import Path
    >>> from pycrumbs import tracked
    >>>
    >>> @tracked(
    ...     literal_directory=Path('/home/user/proj/'),
    ...     subdirectory_name_parameter='model_name',
    ...     include_uuid=True,
    ... )
    >>> def my_train_fun(model_name: str):
    >>>     print(model_name)
    >>>     # Do something...
    >>>     pass
    >>>
    >>> # Record will be placed at, e.g.
    >>> # /home/user/proj/my_model_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2/my_train_fun_record.json
    >>> my_train_fun(model_name='my_model')
    >>> # prints my_model_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2

    Alternatively, you may specify an alternative parameter of the wrapped
    function into which the full updated output directory path will be placed.
    Use the directory_injection_parameter to specify this. This is required when
    you append a timestamp or a UUID without specifying an directory_parameter
    or subdirectory_name_parameter, otherwise there is no way for the wrapped
    function to access the updated output directory. However, you may use
    this at any time for your convenience.

    >>> from pathlib import Path
    >>> from pycrumbs import tracked
    >>>
    >>> @tracked(
    ...     literal_directory=Path('/home/user/proj/'),
    ...     include_uuid=True,
    ...     directory_injection_parameter='model_directory'
    ... )
    >>> def my_train_fun(
    ...     model_directory: Path | None
    ... ):
    >>>     print(model_directory)
    >>>     # Do something...
    >>>     pass
    >>>
    >>> # Record will be placed at, e.g.
    >>> # /home/user/proj_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2/my_train_fun_record.json
    >>> my_train_fun()
    >>> # prints /home/user/proj_2dddbaa6-620f-4aaa-9883-eb3557dbbdb2

    Additionally specify a seed parameter by intercepting the runtime value of
    a parameter of the decorated function. This is always recommended as it
    allows re-running the job with the same seed without having to change the
    code.

    >>> from pathlib import Path
    >>> from pycrumbs import tracked
    >>>
    >>> @tracked(
    ...     literal_directory=Path('/home/user/proj/'),
    ...     subdirectory_name_parameter='model_name',
    ...     seed_parameter='seed'
    ... )
    >>> def my_train_fun(model_name: str, seed: int | None = None):
    >>>     # Do something...
    >>>     print(seed)
    >>>
    >>> # Seed will be injected at runtime by the decorator mechanism,
    >>> # you don't have to do anything else
    >>> my_train_fun(model_name='my_model')
    >>> # prints e.g. 272428
    >>>
    >>> # But you can manually specify the seed later to reproduce, without
    >>> # having to alter the function signature
    >>> my_train_fun(model_name='my_model', seed=272428)
    >>> # prints 272428

    If you use this decorator in combination with decorators from the click
    module (and most other common decorators), you should place this decorator
    last. This is because the click decorators alter the function signature,
    which will break the operation of tracked. The last decorator is
    applied first, meaning that it will operate on the function with its
    original signature as intended.

    >>> from pathlib import Path
    >>> import click
    >>> from pycrumbs import tracked
    >>>
    >>> @click.command()
    >>> @click.argument('model_name')
    >>> @click.option('--seed', '-s', type=int, help='Random seed.')
    >>> @tracked(
    ...     literal_directory=Path('/home/user/proj/'),
    ...     subdirectory_name_parameter='model_name',
    ...     seed_parameter='seed'
    ... )
    >>> def my_train_fun(model_name: str, seed: int | None = None):
    >>>     # Do something...
    >>>     print(seed)

    """  # noqa: E501
    if (literal_directory is None) == (directory_parameter is None):
        raise TypeError(
            'Exactly one of "literal_directory" or "directory_parameter" '
            'must be provided.'
        )
    if include_timestamp and include_uuid:
        raise ValueError(
            'Options "include_timestamp" and "include_uuid" are incompatible.'
        )
    if include_timestamp or include_uuid:
        if (
            directory_injection_parameter is None and  # noqa: W504
            directory_parameter is None and  # noqa: W504
            subdirectory_name_parameter is None
        ):
            option_used = (
                'include_timestamp' if include_timestamp
                else 'include_uuid'
            )
            raise TypeError(
                f'If {option_used} is True and neither directory_parameter '
                'nor subdirectory_name_parameter are specified, a parameter '
                'of the decorated function must be specified via '
                '"directory_injection_parameter" in order for the decorated '
                'function to have access to the dynamically created output '
                'directory.'
            )

    def decorator(function: Callable) -> Callable:

        # Get the function signature check parameters (and use to bind later)
        signature = inspect.signature(function)

        # Check all parameters specified actually exist
        if directory_parameter is not None:
            output_dir_parameter_loc = cast(str, directory_parameter)
            if output_dir_parameter_loc not in signature.parameters:
                raise ValueError(
                    f"No such parameter '{output_dir_parameter_loc}' "
                    "for function '{function.__name__}'."
                )
        if subdirectory_name_parameter is not None:
            if subdirectory_name_parameter not in signature.parameters:
                raise ValueError(
                    f"No such parameters '{subdirectory_name_parameter}' "
                    "for function '{function.__name__}'."
                )
        if directory_injection_parameter is not None:
            if directory_injection_parameter not in signature.parameters:
                raise ValueError(
                    f"No such parameter '{directory_injection_parameter}' "
                    f"for function '{function.__name__}'."
                )
        if seed_parameter is not None:
            if seed_parameter not in signature.parameters:
                raise ValueError(
                    f"No such parameter '{seed_parameter}' for function "
                    f"'{function.__name__}'."
                )

        # Get information about source file to track the function
        try:
            source_file = inspect.getsourcefile(function)
        except TypeError:
            source_file = None

        # Git information about the file in which the function is defined
        if not disable_git_tracking:
            module = inspect.getmodule(function)
            if module is None:
                raise RuntimeError(
                    'Could not determine module for the decorated '
                    'function.'
                )
            git_info = get_git_info(
                module,
                allow_dirty=allow_dirty_repo
            )

        # Git info on other requested modules
        if extra_modules is not None:
            extra_modules_git_info = {
                m if isinstance(m, str) else m.__name__: get_git_info(
                    m,
                    allow_dirty=allow_dirty_repo
                )
                for m in extra_modules
            }

        # Get (static) environment information assumed not to change between
        # invocations
        environment_info = get_environment_info()

        # Get list of packages, assumed not to change between calls
        if include_package_inventory:
            package_inventory = get_installed_packages()

        record_name_local = (
            record_filename or f'{function.__name__}_record'
        )

        # The actual wrapped function
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            record: Dict[str, Any] = {}

            # UUID to identify this run
            job_uuid = uuid4()
            record['uuid'] = str(job_uuid)

            start_time = datetime.datetime.now()
            record['timing'] = {
                'start_time': str(start_time)
            }

            # Add in copy of environment info
            record['environment'] = deepcopy(environment_info)

            # Environment variables could in principle change between different
            # invocations of the function (though probably shouldn't...)
            # To be safe, they are gathered every time
            record['environment'][
                'environment_variables'
            ] = get_environment_vars(
                extra_environment_variables=extra_environment_variables
            )

            if include_package_inventory:
                record['package_inventory'] = package_inventory

            # Information about the called function
            try:
                bound_args = signature.bind(*args, **kwargs)
            except TypeError:
                # If the parameter is marked for injection but was not passed
                # need to manually inject a value (None) here for binding to
                # work
                if directory_injection_parameter is not None:
                    kwargs[directory_injection_parameter] = None
                    bound_args = signature.bind(*args, **kwargs)
                else:
                    raise
            bound_args.apply_defaults()
            record['called_function'] = {
                'name': function.__name__,
                'module': function.__module__,
                'source_file': source_file,
                'parameters': {
                    k: _format_json(v, 200)
                    for k, v in bound_args.arguments.items()
                }
            }

            if not disable_git_tracking:
                record['tracked_module'] = git_info
            if extra_modules is not None:
                record['extra_tracked_modules'] = extra_modules_git_info

            if literal_directory is not None:
                record_dir = Path(literal_directory)
            else:
                # Mypy can't figure out that this must be non-None due to
                # earlier checks
                record_dir = Path(
                    bound_args.arguments[output_dir_parameter_loc]
                )

            if subdirectory_name_parameter is not None:
                record_dir.mkdir(exist_ok=True, parents=create_parents)
                subdir_name = bound_args.arguments[subdirectory_name_parameter]

                record_dir /= subdir_name

            if include_timestamp:
                timestamp = start_time.strftime(_TIMESTAMP_FMT)
                new_name = f'{record_dir.name}_{timestamp}'
                record_dir = record_dir.resolve().with_name(new_name)
            elif include_uuid:
                new_name = f'{record_dir.name}_{job_uuid}'
                record_dir = record_dir.resolve().with_name(new_name)

            seed: Optional[int] = None
            if seed_parameter is not None:
                seed_ = bound_args.arguments[seed_parameter]
                if not isinstance(seed_, int) and seed_ is not None:
                    raise TypeError(
                        "Value of the seed parameter must be 'int' or 'None'"
                        f"but got type {type(seed_)} for parameter "
                        "'{seed_parameter}' of function "
                        "'{function.__name__}'."
                    )
                seed = seed_
            seed = seed_tasks(seed)
            record['seed'] = seed

            # Inject the seed parameter
            if seed_parameter is not None:
                bound_args.arguments[seed_parameter] = seed

            # Inject the final output directory
            if directory_injection_parameter is not None:
                bound_args.arguments[
                    directory_injection_parameter
                ] = record_dir
            elif include_uuid or include_timestamp:
                # Inject the parameter into whatever defined it
                if subdirectory_name_parameter is not None:
                    bound_args.arguments[subdirectory_name_parameter] = (
                        record_dir.name
                    )
                else:
                    if directory_parameter is not None:
                        bound_args.arguments[directory_parameter] = record_dir

            record_dir.mkdir(exist_ok=True, parents=create_parents)
            if require_empty_directory:
                i = record_dir.iterdir()
                try:
                    next(i)
                except StopIteration:
                    # Directory is empty
                    pass
                else:
                    raise FileExistsError(
                        f"Directory {record_dir} is not empty."
                    )

            # Record parameters that are actually used to call the function
            record['called_function']['altered_parameters'] = {
                k: _format_json(v, 200)
                for k, v in bound_args.arguments.items()
            }

            full_record_path = record_dir / record_name_local

            if not full_record_path.name.endswith(".json"):
                full_record_path = full_record_path.with_name(
                    full_record_path.stem + ".json"
                )

            chaining = False
            if chain_records:
                if full_record_path.exists():
                    chaining = True
                    with full_record_path.open("r") as jf:
                        previous_record = json.load(jf)

                    if isinstance(previous_record, List):
                        out_record = previous_record + [record]

                    else:
                        out_record = [previous_record, record]
                else:
                    out_record = record
            else:
                out_record = record

            write_record(full_record_path, record=out_record)

            # Run the function as normal
            result = function(*bound_args.args, **bound_args.kwargs)

            # Record the end time and write out again
            end_time = datetime.datetime.now()
            record['timing']['end_time'] = str(end_time)
            record['timing']['run_time'] = str(end_time - start_time)

            if chaining:
                if isinstance(previous_record, List):
                    out_record = previous_record + [record]

                else:
                    out_record = [previous_record, record]

            else:
                out_record = record
            
            write_record(full_record_path, record=out_record)

            return result

        return wrapper
    return decorator
