# Idpgan

This is the idpGAN repository. IdpGAN is a machine-learning based conformational ensemble generator for coarse grained (CG) models of intrinsically disordered proteins (IDPs).

Details of idpGAN are described in [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1).

## How to run

To use idpGAN (implemented in [PyTorch](https://pytorch.org)), you can run a Jupyter notebook illustrating idpGAN functionalities.

The notebook shows how to use the generator neural network of idpGAN to generate 3D structures of CG IDPs.

There are two ways in which you can run the notebook.

### Colab version

This is the easiest way. Just reach the notebook at: [idpGAN Colab notebook](https://colab.research.google.com/github/feiglab/idpgan/blob/main/notebooks/idpgan_experiments.ipynb).

Running the initial cells will automatically install all the dependencies.

NOTE: please make sure to use a GPU runtime to speed up the idpGAN generation process. If you use the default runtime (running on CPU), generating conformational ensembles with idpGAN could take several minutes. To use a GPU runtime:
  - Use the `Edit` -> `Notebook` settings item on the main menu of the Colab page.
  - Set the `Hardware accelerator` option to `GPU`.

### Running locally

You can also run the notebook on your machine. What you need to do is:
  - Make sure to have NumPy, Matplotlib and PyTorch installed.
    - Optional: if you want to visualize 3D conformational ensembles in Jupyter, also install [NGLview](https://github.com/nglviewer/nglview) and [MDTraj](https://github.com/mdtraj/mdtraj).
  - Clone or [download](https://github.com/feiglab/idpgan/archive/refs/heads/main.zip) this repository on your system.
  - Make sure to have the `idpgan` directory found in the root of this repository in your `PYTHONPATH` so that you can import the `idpgan` library.
  - Run the `idpgan_experiments.ipynb` notebook in the `notebooks` directory.
  - In the notebook, there is the following line of code: `data_dp = "data"`. It should be the path on your system of the `data` directory of this repository (which contains data files and the generator weights). Make sure to edit it so that it points to the right location.

## Datasets

In `data` directory of this repository, we have the following files:
  - `idptest.fasta`: a [FASTA](https://en.wikipedia.org/wiki/FASTA_format) file storing all the sequences of *IDP_test*, the test set of idpGAN.
  - `idpgan_training_set.fasta`: a FASTA file storing all the sequences used in the training set of idpGAN. All of them were obtained from [DisProt](https://disprot.org).
  - `hbval_split_[01234].txt`: files storing the names of the training set sequences used in the five validation partitions of the *HB_val* set in the [idpGAN article](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1).
  
## References

- Janson G, Valdes-Garcia G, Heo L, Feig M. Direct Generation of Protein Conformational Ensembles via Machine Learning.
BioRxiv (2022) doi: [10.1101/2022.06.18.496675](https://www.biorxiv.org/content/10.1101/2022.06.18.496675v1.article-info)


## Contact

[FeigLab](https://feig.bch.msu.edu), mfeiglab@gmail.com

[Michigan State University](https://msu.edu)
