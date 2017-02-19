# BinaryConnect_tf
An unofficial Tensorflow implementation for the paper <a href="http://arxiv.org/abs/1511.00363">BinaryConnect: Training Deep Neural Networks with binary weights during propagations</a>.

# Requirements
* <a href="https://www.tensorflow.org/install/">Tensorflow 1.0</a>
* Python 3.5
* <a href="https://pypi.python.org/pypi/six">six</a>

The code was tested on windows 7/8.1/10.

# Usage
To train a model run [runme_train.py](runme_train.py).

* Toggle binary training using `--binary=True/False`.
* Toggle stochastic training using `--stochastic=True/False`.

CIFAR-10 dataset will be automatically downloaded to `./dataset` folder.

For additional configurations run `python run_me.py -h`.


To test a model run [runme_test.py](runme_test.py).
Pass a trained model path (ckpt file) by setting `--model_path`.
