import os
import sys
import tarfile
import shutil
from six.moves import urllib
import numpy as np


def maybe_download(filename, origin, dst_dir, untar=False):
    """Download and extract the tarball from Alex's website."""

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if untar:
        untar_fpath = os.path.join(dst_dir, filename)
        filepath = untar_fpath + '.tar.gz'
    else:
        filepath = os.path.join(dst_dir, filename)

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(origin, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if not os.path.exists(untar_fpath):
        print('Untaring file...')
        tfile = tarfile.open(filepath, 'r:gz')
        try:
            tfile.extractall(path=dst_dir)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(untar_fpath):
                if os.path.isfile(untar_fpath):
                    os.remove(untar_fpath)
                else:
                    shutil.rmtree(untar_fpath)
            raise
        tfile.close()

    return untar_fpath


def to_categorical(y, nb_classes=None):
    """Converts a class vector (integers) to binary class matrix (one-hot vectors).

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to nb_classes).
        nb_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not nb_classes:
        nb_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, nb_classes))
    categorical[np.arange(n), y] = 1
    return categorical
