"""Tests for the pycrumbs.track module."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import pycrumbs
from pycrumbs import track

import pytest


def reformat_time(time: str) -> str:
    """Reformat time string saved in record to one used in directory name."""
    return datetime.fromisoformat(time).strftime(track._TIMESTAMP_FMT)


def test_records_self_object():
    """Test get_git_info with the pycrumbs module object."""
    record = pycrumbs.get_git_info(pycrumbs)
    assert 'git_commit_hash' in record
    assert 'git_active_branch' in record
    assert 'git_is_dirty' in record


def test_records_self_name():
    """Test get_git_info with the pycrumbs module by name."""
    record = pycrumbs.get_git_info('pycrumbs')
    assert 'git_commit_hash' in record
    assert 'git_active_branch' in record
    assert 'git_is_dirty' in record


def test_records_standard_library():
    """Test get_git_info with a standard library module."""
    with pytest.raises(RuntimeError):
        pycrumbs.get_git_info('os')


def test_auto_wrap_literal():
    """Test tracked with a literal output path."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp)
        def some_fun(x):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {'x': 4}
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_literal_with_timestamp():
    """Test tracked with a literal output path with timestamp."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        subdir = temp / 'subdir'

        @pycrumbs.tracked(
            literal_directory=subdir,
            include_timestamp=True,
            directory_injection_parameter='literal_directory',
        )
        def some_fun(x, literal_directory=None):
            assert isinstance(literal_directory, Path)
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = outdir.joinpath('some_fun_record.json')
        assert saved_record.exists()
        assert outdir.name.startswith('subdir_')
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {
            'x': 4,
            'literal_directory': None
        }
        assert record_data['called_function']['altered_parameters'] == {
            'x': 4,
            'literal_directory': repr(outdir)
        }
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        expected_time = reformat_time(record_data['timing']['start_time'])
        assert outdir.name.endswith(expected_time)
        assert 'end_time' in record_data['timing']


def test_auto_wrap_literal_with_uuid():
    """Test tracked with a literal output path and a uuid."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir = temp / 'subdir'

        @pycrumbs.tracked(
            literal_directory=subdir,
            include_uuid=True,
            directory_injection_parameter='literal_directory',
        )
        def some_fun(x, literal_directory=None):
            assert isinstance(literal_directory, Path)
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = outdir.joinpath('some_fun_record.json')
        assert saved_record.exists()
        assert outdir.name.startswith('subdir_')
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {
            'x': 4,
            'literal_directory': None
        }
        assert record_data['called_function']['altered_parameters'] == {
            'x': 4,
            'literal_directory': repr(outdir)
        }
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert outdir.name.endswith(record_data['uuid'])
        assert 'end_time' in record_data['timing']


def test_auto_wrap_literal_injecting_required_parameter():
    """Test tracked when injecting into a required parameter."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir = temp / 'subdir'

        @pycrumbs.tracked(
            literal_directory=subdir,
            include_uuid=True,
            directory_injection_parameter='literal_directory',
        )
        def some_fun(x, literal_directory):
            assert isinstance(literal_directory, Path)
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = outdir.joinpath('some_fun_record.json')
        assert saved_record.exists()
        assert outdir.name.startswith('subdir_')
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {
            'x': 4,
            'literal_directory': None
        }
        assert record_data['called_function']['altered_parameters'] == {
            'x': 4,
            'literal_directory': repr(outdir)
        }
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert outdir.name.endswith(record_data['uuid'])
        assert 'end_time' in record_data['timing']


def test_auto_wrap_literal_subdir():
    """Test tracked with a sub-directory."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir_name = 'my_model'

        @pycrumbs.tracked(
            literal_directory=temp,
            subdirectory_name_parameter='model_name'
        )
        def some_fun(x, model_name):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, model_name=subdir_name)
        assert y == 5
        saved_record = temp.joinpath(subdir_name, 'some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'model_name': subdir_name}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_dir_param_with_timestamp():
    """Test tracked with a directory parameter and a timestamp."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir_name = 'my_model'

        @pycrumbs.tracked(
            directory_parameter='literal_directory',
            include_timestamp=True,
        )
        def some_fun(x, literal_directory):
            assert literal_directory.name.startswith(subdir_name)
            assert len(literal_directory.name) > len(subdir_name)
            return x + 1

        initial_output_dir = temp / subdir_name
        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, literal_directory=initial_output_dir)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = temp.joinpath(outdir, 'some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'literal_directory': repr(initial_output_dir)}
        assert record_data['called_function']['parameters'] == params_dict
        alt_dict = {
            'x': 4,
            'literal_directory': repr(outdir),
        }
        assert (
            record_data['called_function']['altered_parameters'] == alt_dict
        )
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']
        assert outdir.name.startswith(subdir_name + '_')


def test_auto_wrap_subdir_with_timestamp():
    """Test tracked with a sub-directory and a timestamp."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir_name = 'my_model'

        @pycrumbs.tracked(
            literal_directory=temp,
            subdirectory_name_parameter='model_name',
            include_timestamp=True,
        )
        def some_fun(x, model_name):
            assert model_name.startswith(subdir_name)
            assert len(model_name) > len(subdir_name)
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, model_name=subdir_name)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = temp.joinpath(outdir, 'some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'model_name': subdir_name}
        assert record_data['called_function']['parameters'] == params_dict
        alt_dict = {
            'x': 4,
            'model_name': outdir.name,
        }
        assert (
            record_data['called_function']['altered_parameters'] == alt_dict
        )
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']
        assert outdir.name.startswith(subdir_name + '_')


def test_auto_wrap_subdir_with_timestamp_inject():
    """Test tracked with a sub-directory and a timestamp."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir_name = 'my_model'

        @pycrumbs.tracked(
            literal_directory=temp,
            subdirectory_name_parameter='model_name',
            include_timestamp=True,
            directory_injection_parameter='literal_directory',
        )
        def some_fun(x, model_name, literal_directory=None):
            assert isinstance(literal_directory, Path)
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, model_name=subdir_name)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = temp.joinpath(outdir, 'some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'model_name': subdir_name, 'literal_directory': None}
        assert record_data['called_function']['parameters'] == params_dict
        alt_dict = {
            'x': 4,
            'model_name': subdir_name,
            'literal_directory': repr(outdir)
        }
        assert (
            record_data['called_function']['altered_parameters'] == alt_dict
        )
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']
        assert outdir.name.startswith(subdir_name + '_')
        expected_time = reformat_time(record_data['timing']['start_time'])
        assert outdir.name.endswith(expected_time)


def test_auto_wrap_subdir_with_uuid():
    """Test tracked with a sub-directory and a uuid."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir_name = 'my_model'

        @pycrumbs.tracked(
            literal_directory=temp,
            subdirectory_name_parameter='model_name',
            include_uuid=True,
        )
        def some_fun(x, model_name):
            assert model_name.startswith(subdir_name)
            assert len(model_name) > len(subdir_name)
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, model_name=subdir_name)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = temp.joinpath(outdir, 'some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'model_name': subdir_name}
        assert record_data['called_function']['parameters'] == params_dict
        alt_dict = {
            'x': 4,
            'model_name': outdir.name,
        }
        assert (
            record_data['called_function']['altered_parameters'] == alt_dict
        )
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']
        assert outdir.name.startswith(subdir_name + '_')
        assert outdir.name.endswith(record_data['uuid'])


def test_auto_wrap_subdir_with_uuid_and_inject():
    """Test tracked with a sub-directory and a uuid."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir_name = 'my_model'

        @pycrumbs.tracked(
            literal_directory=temp,
            subdirectory_name_parameter='model_name',
            include_uuid=True,
            directory_injection_parameter='literal_directory',
        )
        def some_fun(x, model_name, literal_directory=None):
            assert isinstance(literal_directory, Path)
            assert model_name == subdir_name
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, model_name=subdir_name)
        assert y == 5
        outdir = next(temp.iterdir())
        saved_record = temp.joinpath(outdir, 'some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'model_name': subdir_name, 'literal_directory': None}
        assert record_data['called_function']['parameters'] == params_dict
        alt_dict = {
            'x': 4,
            'model_name': subdir_name,
            'literal_directory': repr(outdir)
        }
        assert (
            record_data['called_function']['altered_parameters'] == alt_dict
        )
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']
        assert outdir.name.startswith(subdir_name + '_')
        assert outdir.name.endswith(record_data['uuid'])


def test_auto_wrap_literal_record_name():
    """Test tracked with an alternative record name."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        record_filename = 'my_record'

        @pycrumbs.tracked(literal_directory=temp, record_filename=record_filename)
        def some_fun(x):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        saved_record = temp.joinpath(f'{record_filename}.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {'x': 4}
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_arg_extra_modules_string():
    """Test tracked with extra modules as strings."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(
            literal_directory=temp,
            extra_modules=['pycrumbs']
        )
        def some_fun(x):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {'x': 4}
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_arg_extra_modules_object():
    """Test tracked with extra modules as objects."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(
            literal_directory=temp,
            extra_modules=[pycrumbs]
        )
        def some_fun(x):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 5
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        assert record_data['called_function']['parameters'] == {'x': 4}
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_arg():
    """Test tracked intercepting output location parameter as an arg."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(directory_parameter='path')
        def some_fun(x, path):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, temp)
        assert y == 5
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'path': repr(temp)}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_kwarg():
    """Test tracked intercepting output location parameter as a kwarg."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(directory_parameter='path')
        def some_fun(x, path):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, path=temp)  # 'path' is a kwarg
        assert y == 5
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'path': repr(temp)}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_default():
    """Test tracked intercepting output location as a default arg."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(directory_parameter='path')
        def some_fun(x, path=temp):
            return x + 1

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)  # 'path' takes default value
        assert y == 5
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'path': repr(temp)}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_args_sig():
    """Test tracked with a function defined with *args."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp)
        def some_fun(*args):
            return args

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4, 5, 6)
        assert y == (4, 5, 6)
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'args': [4, 5, 6]}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_args_kwargs_sig():
    """Test tracked with a function defined with *args."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp)
        def some_fun(*args, **kwargs):
            return args, kwargs

        assert some_fun.__name__ == 'some_fun'
        out_args, out_kwargs = some_fun(4, 5, thing='frog')
        assert out_args == (4, 5)
        assert out_kwargs == {'thing': 'frog'}
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'args': [4, 5], 'kwargs': {'thing': 'frog'}}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_seed():
    """Test tracked with a specified seed."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        seed = 98

        @pycrumbs.tracked(literal_directory=temp, seed_parameter='seed')
        def some_fun(x, seed):
            return x + seed

        assert some_fun.__name__ == 'some_fun'
        # Test seed as an arg and a kwarg
        y = some_fun(4, seed)
        assert y == 4 + seed
        y = some_fun(4, seed=seed)
        assert y == 4 + seed
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'seed': seed}
        assert record_data['called_function']['parameters'] == params_dict
        assert record_data['seed'] == seed
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_seed_none():
    """Test tracked with an empty seed."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp, seed_parameter='seed')
        def some_fun(x, seed=None):
            return seed

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4, 'seed': None}
        assert record_data['called_function']['parameters'] == params_dict
        assert isinstance(record_data['seed'], int)
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']

        # Check the seed was insert correctly into the called function
        assert y == record_data['seed']


def test_auto_wrap_disabled_tracking():
    """Test tracked with no git tracking."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp, disable_git_tracking=True)
        def some_fun(x):
            return x

        assert some_fun.__name__ == 'some_fun'
        y = some_fun(4)
        assert y == 4
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': 4}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']
        assert 'tracked_module' not in record_data


def test_auto_wrap_non_serializable_param():
    """Test tracked with a non-serializable parameter."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp)
        def some_fun(x):
            return x

        assert some_fun.__name__ == 'some_fun'
        x = uuid4()  # non-serializable
        y = some_fun(x)
        assert y == x
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': repr(x)}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_auto_wrap_long_param():
    """Test tracked with a long parameter."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp)
        def some_fun(x):
            return x

        assert some_fun.__name__ == 'some_fun'
        x = list(range(1000))  # too long
        y = some_fun(x)
        assert y == x
        saved_record = temp.joinpath('some_fun_record.json')
        assert saved_record.exists()
        with saved_record.open('r') as jf:
            record_data = json.load(jf)
        assert record_data['called_function']['name'] == 'some_fun'
        params_dict = {'x': '<suppressed due to excessive length>'}
        assert record_data['called_function']['parameters'] == params_dict
        assert 'seed' in record_data
        assert 'start_time' in record_data['timing']
        assert 'end_time' in record_data['timing']


def test_require_empty_success():
    """Test tracked with require_empty_directory is True."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()

        @pycrumbs.tracked(literal_directory=temp, require_empty_directory=True)
        def some_fun(x):
            pass

        some_fun(0)


def test_require_empty_fail():
    """Test tracked with require_empty_directory True when non-empty'."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        temp_content = temp / 'junk'
        temp_content.touch()

        @pycrumbs.tracked(literal_directory=temp, require_empty_directory=True)
        def some_fun(x):
            pass

        with pytest.raises(FileExistsError):
            some_fun(0)


def test_parents():
    """Test tracked with create_parents is True."""
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp).resolve()
        subdir = temp / 'subdir'

        @pycrumbs.tracked(literal_directory=subdir, create_parents=True)
        def some_fun(x):
            pass

        some_fun(0)
        assert subdir.exists()
        assert subdir.joinpath('some_fun_record.json').exists()
