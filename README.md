# Idpgan

This is the idpGAN repository. IdpGAN is a machine-learning based conformational ensemble generator for coarse grained (CG) models of intrinsically disordered proteins (IDPs).

Details of idpGAN are described in [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1).

## How to run

To use idpGAN (implemented in [PyTorch](https://pytorch.org)), you can run a Jupyter notebook illustrating idpGAN functionalities.

The notebook shows how to use the generator neural network of idpGAN to generate 3D structures of CG IDPs.

There are two ways in which you can run the notebook.

### Colab version

This is the easiest way. This option allows you to run the notebook remotely. Just reach the notebook at: [idpGAN Colab notebook](https://colab.research.google.com/github/feiglab/idpgan/blob/main/notebooks/idpgan_experiments.ipynb).

Running the initial cells will automatically install all the dependencies (please note that the installation may require a few minutes).

NOTE: make sure to use a GPU runtime to largely speed up the idpGAN conformation generation process. If you use the default runtime (running on CPU), the process could take several minutes. To use a GPU runtime:
  - Use the `Edit` -> `Notebook settings` item in the main menu of the Colab page.
  - Set the `Hardware accelerator` option to `GPU`.

### Running locally

You can also run the notebook on your machine. What you need to do is:
  - Make sure to have NumPy, Matplotlib and PyTorch installed.
    - Optional: if you want to visualize 3D conformational ensembles in Jupyter, also install [NGLview](https://github.com/nglviewer/nglview) and [MDTraj](https://github.com/mdtraj/mdtraj).
  - Clone or [download](https://github.com/feiglab/idpgan/archive/refs/heads/main.zip) this repository on your system.
  - Make sure to have the `idpgan` directory (the one found in the root of this repository) in your [`PYTHONPATH`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH), so that you can import the `idpgan` library.
  - Run the `idpgan_experiments.ipynb` notebook in the `notebooks` directory.
  - In the notebook, there is the following line of code: `data_dp = "data"`. It must point to the path on your system where the `data` directory of this repository is located (the directory contains data files and the generator weights). Make sure to edit the line so that it points to the right location.

## Datasets

In the `data` directory of this repository, we have the following files with information on the training, validation and test sets of idpGAN:
  - `idpgan_training_set.fasta`: a FASTA file storing all the sequences used in the training set of idpGAN. All of them were obtained from [DisProt](https://disprot.org).
  - `hbval_split_[01234].txt`: files storing the names of the training set sequences used in the five validation partitions of the *HB_val* set in the [idpGAN article](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1).
  - `idptest.fasta`: a [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file storing all the sequences of *IDP_test*, the test set of idpGAN.

We also have the following files, that allow you to run a demo of the generatative model on a small dataset:
  - `generator.pt`: PyTorch weights for a pre-trained generator model. They are the same weights we used to generate conformations for the *IDP_test* proteins in the article.
  - `*.npy`: NumPy array files storing the xyz coordinates for 5 x 1000 ns MD simulations for two *IDP_test* proteins and one poly-alanine molecule.
  
## System requirements

You can run the idpGAN notebook [remotely on Colab](https://colab.research.google.com/github/feiglab/idpgan/blob/main/notebooks/idpgan_experiments.ipynb). You do not need to install any software on your system, you only need to first login on your Google account.

Otherwise you can run the idpGAN notebook (and library) locally on your machine. Any Python (version >= 3.8) with the [above requirements](#running-locally) should work well on all major operating systems (Windows, Mac, Linux). We developed and tested the code on Linux, Python (3.8.10) and PyTorch (1.7.1).

In all cases, if you plan to generate conformational ensembles with large number of snapshots (> 5000), we suggest to use [PyTorch GPU support](https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk) to largely speed up computational times.
  
## References

- Janson G, Valdes-Garcia G, Heo L, Feig M. Direct Generation of Protein Conformational Ensembles via Machine Learning.
BioRxiv (2022) doi: [10.1101/2022.06.18.496675](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1.article-info)

## Contact

[FeigLab](https://feig.bch.msu.edu), mfeiglab@gmail.com

[Michigan State University](https://msu.edu)
