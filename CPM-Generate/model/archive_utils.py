"""Restricted model-archive extraction, including on legacy Python versions."""

import ntpath
import os
import tarfile


def extract_model_archive(archive, destination):
    """Extract regular files/directories only, after validating every member.

    The caller must supply a fresh private temporary directory. Model archives
    do not need links or special files. This does not make pickle model weights
    safe to load, or impose archive memory/disk quotas: use trusted models only.
    """
    root = os.path.realpath(destination)
    members = archive.getmembers()
    for member in members:
        name = member.name
        if (not name or "\x00" in name or "\\" in name or
                ntpath.splitdrive(name)[0] or ntpath.isabs(name) or
                ".." in name.split("/")):
            raise ValueError("Unsafe model archive path: {!r}".format(name))
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
            raise ValueError("Unsupported model archive member: {!r}".format(name))
        target = os.path.realpath(os.path.join(root, name))
        if os.path.commonpath([root, target]) != root:
            raise ValueError("Model archive path escapes destination: {!r}".format(name))
    # No archive-created symlink/hardlink can invalidate the checks above.
    archive.extractall(root, members=members)
