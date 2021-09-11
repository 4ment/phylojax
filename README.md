# phylojax

## Installation

### Get the phylojax source
```bash
git clone https://github.com/4ment/phylojax
cd phylojax
```

### Install dependencies

Installing dependencies using Anaconda
```bash
conda env create -f environment.yml
conda activate phylojax
```

### Install phylotorch
```bash
python setup.py install
```

### Check install
```bash
phylojax --help
```

### Quick start
```bash
phylojax -t examples/fluA.tree -i examples/fluA.fa -m JC69
```