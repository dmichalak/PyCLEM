import os
import re
from pathlib import Path
from typing import List, Tuple, Union


def get_files(pattern: str, subdir: Union[str, Path] = Path(),
              id_dict: bool = False) -> Tuple[List[str], List[Tuple]]:
    r"""Get all files matching a regular expression

    Author: Lukas Schrangl (TU Wien)

    Parameters
    ----------
    pattern
        Regular expression to search in the file name. Search is performed
        on the path relative to `subdir`. One can also define groups (using
        parenthesis), which will be returned in addition to the matching
        file names. **A note to Windows users: Use a forward slash (/) for
        subdirectories.**
    subdir
        Any regular expression matching will be performed relative to `subdir`.
    id_dict
        If `True`, return IDs as a dict. Only works for named groups in
        `pattern`.

    Returns
    -------
    Sorted list of file where `pattern` could be matched. as well as values of
    the groups defined in the `pattern`. Values are converted to int if
    possible, otherwise conversion to float is attempted. If that fails as
    well, the string is used.

    Examples
    --------
    >> names, ids = get_files(r'^image_(.*)_(\d{3}).tif$', 'subdir')
    >> names
    ['image_xxx_001.tif', 'image_xxx_002.tif', 'image_yyy_003.tif']
    >> ids
    [('xxx', 1), ('xxx', 2), ('yyy', 3)]
    """
    r = re.compile(pattern)
    flist = []
    idlist = []
    for dp, dn, fn in os.walk(subdir):
        reldir = Path(dp).relative_to(subdir)
        for f in fn:
            relp = (reldir / f).as_posix()
            m = r.search(relp)
            if m is None:
                continue
            # For compatibility, append path as string.
            # However, one could simply append reldir / f
            flist.append(relp)
            if id_dict:
                ids = {k: _conv_value(v) for k, v in m.groupdict().items()}
            else:
                ids = tuple(_conv_value(v) for v in m.groups())
            idlist.append(ids)
    slist = sorted(zip(flist, idlist), key=lambda x: x[0])
    return [s[0] for s in slist], [s[1] for s in slist]


def _conv_value(v: str) -> Union[int, float, str]:
    """Try converting value to int, then float, otherwise return unchanged

    Author: Luakas Schrangl (TU Wien)

    Helper function for :py:func:`get_files`

    Parameters
    ----------
    v
        Value for attempted type conversion

    Returns
    -------
    Value as int, if possible, otherwise as float, if possible, otherwise
    unchanged as string.
    """
    for conv in int, float:
        try:
            v = conv(v)
        except ValueError:
            continue
        else:
            break
    return v


def chdir(path):
    """Context manager to temporarily change the working directory

    Author: Lukas Schrangl (TU Wien)

    Parameters
    ----------
    path : str
        Path of the directory to change to. :py:func:`os.path.expanduser` is
        called on this.

    Examples
    --------
    >> with chdir("subdir"):
    ...     # here the working directory is "subdir"
    >> # here we are back
    """
    old_wd = os.getcwd()
    os.chdir(os.path.expanduser(str(path)))
    try:
        yield
    finally:
        os.chdir(old_wd)


def get_subdir(root_dir: Union[str, Path] = Path()) -> List[str]:
    """
    Returns a list of subdirectories in the given root directory.

    Parameters
    ----------
    root_dir : Union[str, Path], optional
        The root directory path. Defaults to the current working directory.

    Returns
    -------
    List[str]
        A sorted list of subdirectories in the specified root directory.

    Examples
    --------
    >> get_subdir(Path('/my/root/directory'))
    ['subdir1', 'subdir2', 'subdir3']

    >> get_subdir()
    ['dir_a', 'dir_b', 'dir_c']
    """
    dirs = []
    for file in os.listdir(root_dir):
        d = os.path.join(root_dir, file)
        if os.path.isdir(d):
            dirs.append(file)
    return dirs


def extract_files(pattern: str, parent: Union[str, Path] = Path()) -> None:
    """
    Finds all files matching the specified pattern and moves them into the given parent folder.

    Parameters
    ----------
    pattern : str
        Regular expression pattern to search for in the file names.
    parent : Union[str, Path], optional
        The target parent folder where matching files will be moved. Defaults to the current working directory.

    Returns
    -------
    None
        This function does not return any value but moves matching files to the specified parent folder.

    Examples
    --------
    >> extract_files(r'image_(.*)_(\d{3}).tif', parent='/target/folder/')
    # Moves files to '/target/folder/'.
    """
    files, _ = get_files(pattern=pattern, subdir=parent, id_dict=False)
    for file in files:
        _, base_fn = os.path.split(file)
        old_name = Path(parent, file)
        new_name = Path(parent, base_fn)
        os.rename(old_name, new_name)

        
def remove_duplicate_files_from_list(files: List[Union[str, Path]]) -> List[Path]:
    """
    Removes duplicates from a list of files. Only the base filename is considered for this meaning if there are files
    with the same name in different subfolders, all but one will be removed from the list.
    :param files: list of filenames (str or Path objects)
    :return: list of filenames (str or Path objects)
    """
    unique_files = set()
    duplicate_files = []
    for file in files:
        # Make sure file is a Path() object
        file = Path(file)
        # Get the filename without the path
        file_name = file.name
        # Check if the filename is already in the set of unique files
        if file_name in unique_files:
            duplicate_files.append(file)
        else:
            unique_files.add(file_name)
    # Remove the duplicate files from the original list
    for duplicate_file in duplicate_files:
        files.remove(duplicate_file)

    return files


if __name__ == '__main__':
    pass

