# A Codebase for Multimodal VAEs

This is the code for my Bachelor-Thesis 'A Codebase for Multimodal VAEs'.

Environment.yml can be used to create the same conda environment that was used by me.
```bash
conda env create -f environment.yml  # create conda env
conda activate gpu_env               # activate conda env
```
Then same as for the old codebase https://github.com/thomassutter/MoPoE
download the data, inception network, and pretrained classifiers:
```bash
curl -L -o tmp.zip https://drive.google.com/drive/folders/1lr-laYwjDq3AzalaIe9jN4shpt1wBsYM?usp=sharing
unzip tmp.zip
unzip data_mnistsvhntext.zip -d data/
unzip PolyMNIST.zip -d data/
```

## Experiments

Experiments can be started by running the respective `job_*` script.
To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the config's `METHOD` variabe to "poe", "moe", or "joint\_elbo"
respectively.  By default, each experiment uses `METHOD="joint_elbo"`.

### running MNIST-SVHN-Text
```bash
./job_mnistsvhntext
```

### running PolyMNIST
```bash
./job_mmnist
```
