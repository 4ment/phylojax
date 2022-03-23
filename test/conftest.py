import os

import pytest

from phylojax.substitution import JC69

data_dir = 'examples'


@pytest.fixture
def hello_tree_file():
    return os.path.join(data_dir, 'hello.nwk')


@pytest.fixture
def hello_fasta_file():
    return os.path.join(data_dir, 'hello.fasta')


@pytest.fixture
def flu_a_tree_file():
    return os.path.join(data_dir, 'fluA.tree')


@pytest.fixture
def flu_a_fasta_file():
    return os.path.join(data_dir, 'fluA.fa')


@pytest.fixture
def jc69_model():
    return JC69()
