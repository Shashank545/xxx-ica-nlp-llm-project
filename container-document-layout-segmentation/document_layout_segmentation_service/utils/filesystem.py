import base64
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple, Union

from utils.logger import FusionServicesLogger

fs_logger = FusionServicesLogger('FS', logging.DEBUG)

UMASK_PERMISSION = stat.S_IRWXG | stat.S_IRWXO


def make_folders(
    parent_path: PosixPath, folder_names: Tuple[str]
) -> List[Path]:
    """
    Generic utility function to make folders

    Args:
        parent_path (PosixPath):
            Path of the parent folders
        folder_names (Tuple):
            Tuple of names of the folders to create
    Returns:
        List[Path]:
            List of directory paths
    """
    folder_paths = list()
    try:
        saved_umask = os.umask(UMASK_PERMISSION)
        for folder in folder_names:
            folder_path = Path(str(parent_path), folder)
            fs_logger.debug(f'Creating {folder_path}')
            folder_path.mkdir(exist_ok=True, parents=True)
            folder_paths.append(folder_path)
        os.umask(saved_umask)
    except Exception as error:
        fs_logger.exception('Failed to create working paths')
        raise IOError from error
    return folder_paths


def copy_fileobj_to_storage(
    file: tempfile._TemporaryFileWrapper, destination: PosixPath
) -> None:
    """
    Generic utility to copy file object to file storage destination
    Args:
        file (tempfile._TemporaryFileWrapper):
            File in a TemporaryFile format to be copied
        destination (PosixPath):
            Destination path for file to be copied to
    """
    try:
        saved_umask = os.umask(UMASK_PERMISSION)
        with open(destination, 'wb') as outfile:
            fs_logger.debug(f'Copying {file.filename} to path {destination}')
            shutil.copyfileobj(file.file, outfile)
            os.umask(saved_umask)
    except Exception as error:
        fs_logger.exception(
            f'Failed to copy {file.filename} to path {destination}'
        )
        raise IOError from error


def base64_conversion(filepath: PosixPath) -> str:
    """
    Generic utility to convert image file to base64 string
    Args:
        filepath (PosixPath):
            Path to PNG image in the file system
    Returns:
        str:
           Base64 string
    """
    result = ''
    try:
        fs_logger.debug(f'Converting PNG thumbnail image from path {filepath}')
        saved_umask = os.umask(UMASK_PERMISSION)
        with open(filepath, "rb") as img:
            image_str = base64.b64encode(img.read())
            result = 'data:image/png;base64,' + image_str.decode("utf-8")
            os.umask(saved_umask)
    except Exception as error:
        raise IOError from error
    else:
        fs_logger.debug(
            f'Successfully converted PNG thumbnail image from path {filepath}'
        )

    return result


def get_results_from_path(
    results_path: PosixPath,
) -> Dict[str, Union[bytes, str]]:
    """
    Generic utility to load files in the results directory
    Args:
        results_path (PosixPath):
            Path to the results of the conversion processes
    Returns:
        Dict:
           Results of loaded path
    """
    results: Dict[str, Union[bytes, str]] = {}
    if results_path.exists():
        try:
            saved_umask = os.umask(UMASK_PERMISSION)
            for filepath in results_path.iterdir():
                if filepath.suffix == '.pdf':
                    with open(filepath, 'rb') as pdffile:
                        results['filedata'] = pdffile.read().decode(
                            'utf8', 'replace'
                        )
                if filepath.suffix == '.png':
                    results['thumbnail'] = base64_conversion(filepath)
            os.umask(saved_umask)
        except Exception as error:
            fs_logger.exception(error)
            raise IOError from error
    return results


def ephemeral_storage_cleanup(source: PosixPath) -> None:
    """
    Generic directory removing function
    Args:
        source (PosixPath):
            Source path object
    """
    fs_logger.debug(f'Starting ephemeral storage clean up at {source}')
    saved_umask = os.umask(UMASK_PERMISSION)
    try:
        shutil.rmtree(source)
        fs_logger.debug(f'Completed ephemeral storage clean up at {source}')
        os.umask(saved_umask)
    except Exception as error:
        fs_logger.exception(
            'Failed to remove directories and files from '
            f'ephemeral storage at {source}'
        )
        raise IOError from error
