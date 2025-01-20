import re
import requests
import importlib
from pathlib import Path
from shutil import copyfile, rmtree
from zipfile import ZipFile

import dommel_library.datasets
from dommel_library.datasets.dataset import retry, ConcatDataset

import logging

logger = logging.getLogger(__name__)


def dataset_factory(
    location=None,
    destination=None,
    type="FilePool",
    sequence_length=-1,
    keys=None,
    transform=None,
    clean=False,
    keep_zip=False,
    backend=None,
    **kwargs,
):
    """
    Factory method for the creation of dommel datasets.
    :param location: Url or directory to load from. If None, will initialize
    an empty DictPool.
    :param type: String representing the dataset type to construct
    :param destination: If the dataset needs to copied to another location,
    or downloaded to a certain destination, provide target path here
    :param keys: List of sensor keys to load. Defaults to all keys.
    :param sequence_length: Length of subsequences.
    :param transform: List of data transforms.
    :param clean: Force re-download.
    :param keep_zip: Keep the .zip file after extraction.
    :param backend: parameter to register the backend for the dataset class
    :param kwargs: The parameters for the construction of the pools.
    """
    if isinstance(location, list):
        datasets = []
        for loc in location:
            d = dataset_factory(
                loc,
                destination,
                type,
                sequence_length,
                keys,
                transform,
                clean,
                keep_zip,
                **kwargs,
            )
            datasets.append(d)
        return ConcatDataset(datasets)

    if location:
        if "http" in location[:4]:
            location = download(location, destination, clean)

        if ".zip" in location[-4:]:
            location = unzip(location, destination, keep_zip)

        if destination and not location.startswith(destination):
            location = copy(location, destination)

        kwargs["directory"] = location

    kwargs["sequence_length"] = sequence_length
    kwargs["keys"] = keys

    if not backend:
        backend_module = dommel_library.datasets  # noqa: F
    else:
        backend_module = importlib.import_module(backend)  # noqa: F

    module = f"backend_module.{type}(**{repr(kwargs)}," f"transform=transform)"
    return eval(module)


@retry
def copy(file_location, destination=None):
    """
    Initializes the file transform
    :param file_location: Original location of the dataset.
    :param destination: Destination of the dataset, defaults to
    /tmp/data.
    """
    file_location = Path(file_location)
    if destination:
        destination = Path(destination)
    else:
        destination = Path("/tmp/data")
    # make sure destination exists
    destination.mkdir(parents=True)
    for file in file_location.iterdir():
        dest = destination / file.parts[-1]
        copyfile(file, dest)
    return str(destination)


def unzip(zip_location, destination=None, keep_zip=False):
    """
    Initializes the unzip transform file transform
    :param zip_location: Location of the zip file
    :param destination: Unzip destination. Defaults to /tmp/data
    :param keep_zip: keep the .zip file after extraction
    """
    logger.info("Unzipping dataset %s", zip_location)
    zip_location = Path(zip_location)

    if destination:
        destination = Path(destination)
    else:
        destination = Path("/tmp/data")
    # make sure destination exists
    destination.mkdir(parents=True, exist_ok=True)
    target_dir = destination / zip_location.name[:-4]

    with ZipFile(zip_location, "r") as zip_obj:
        needs_extraction = False
        for name in zip_obj.namelist():
            if not (target_dir / name).exists():
                needs_extraction = True

        if needs_extraction:
            zip_obj.extractall(target_dir)

    dataloc = target_dir

    # if the .zip contains a root dir with the same name
    # as the .zip name, we'll have it twice in the path
    # in that case, just move the target one up
    subdir = dataloc / zip_location.name[:-4]
    if subdir.exists():
        p = subdir.absolute()
        parent_dir = subdir.parents[1]
        tmp_p = parent_dir / (p.name + "_tmp")
        p.rename(tmp_p)
        tmp_p.rename(parent_dir / p.name)

    logger.info("Unzipped to %s", dataloc)

    if not keep_zip:
        zip_location.unlink()

    return str(dataloc)


def download(remote_location, destination=None, clean=False):
    """
    Initializes the file transform
    :param remote_location: URL to the dataset. Must end with (/download)
    to be valid.
    :param destination: Destination of the download operation. Defaults to
    /tmp/data.
    :param clean: Force redownload
    """
    if "http" not in remote_location[:4]:
        raise AssertionError("Invalid url")
    if destination:
        destination = Path(destination)
    else:
        destination = Path("/tmp/data")
    destination.mkdir(parents=True, exist_ok=True)

    with requests.get(remote_location, stream=True) as r:
        content_type = r.headers["content-type"]
        content_disposition = r.headers["content-disposition"]
        file_name = None
        if content_disposition is not None:
            file_name = get_filename_from_cd(content_disposition)

        if file_name is None:
            extension = content_type.split("/")[-1]
            file_name = f"file.{extension}"

        # if the target unzip directory exists, return this
        target_dir = destination / file_name[:-4]
        if target_dir.exists():
            if clean:
                rmtree(target_dir, ignore_errors=True)
            else:
                return str(target_dir)

        # if the .zip exists, return this
        destination /= file_name
        if destination.exists():
            if clean:
                destination.unlink()
            else:
                return str(destination)

        logger.info("Downloading dataset %s", remote_location)
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        logger.info("Download saved at %s", destination)

    return str(destination)


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall("filename=(.+)", cd)
    if len(fname) == 0:
        return None

    fname = fname[0]
    # strip quotes if exist
    if '"' in fname:
        fname = fname.strip('"')

    return fname
