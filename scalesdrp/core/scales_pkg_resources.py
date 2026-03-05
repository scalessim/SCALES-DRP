import importlib_resources
import os

def get_resource_path(pkg, path):
    '''Get the full path to a resource file within a package.'''
    full_path = importlib_resources.files(pkg) / path

    if os.path.exists(full_path):
        return full_path
    else:
        raise FileNotFoundError("Resource file %s not found in package %s" % path, pkg)
