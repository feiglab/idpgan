import numpy as np


def parse_fasta_seq(fasta_fp):
    """Gets the sequence in a one-entry FASTA file."""
    seq = ""
    with open(fasta_fp, "r") as i_fh:
        content = i_fh.read()
        entry_count = content.count(">")
        if entry_count > 1:
            raise ValueError("Can only read FASTA files with one entry.")
        elif entry_count == 0:
            raise ValueError("No entry found in the input file.")
        for line in content.split("\n"):
            if line.startswith(">"):
                continue
            seq += line.rstrip()
    return seq


one_to_three = {"Q": "GLN", "W": "TRP", "E": "GLU", "R": "ARG", "T": "THR",
                "Y": "TYR", "I": "ILE", "P": "PRO", "A": "ALA", "S": "SER",
                "D": "ASP", "F": "PHE", "G": "GLY", "H": "HIS", "K": "LYS",
                "L": "LEU", "C": "CYS", "V": "VAL", "N": "ASN", "M": "MET"}

def seq_to_cg_pdb(seq, out_fp=None):
    """
    Gets an amino acid sequence and returns a template
    CG PDB file.
    """
    pdb_lines = []
    for i, aa_i in enumerate(seq):
        res_idx = i + 1
        aa_i = one_to_three[aa_i]
        line_i = "ATOM{:>7} CG   {} A{:>4}       0.000   0.000   0.000  1.00  0.00\n".format(
                   str(res_idx), aa_i, str(res_idx))
        pdb_lines.append(line_i)
    pdb_content = "".join(pdb_lines)
    if out_fp is not None:
        with open(out_fp, "w") as o_fh:
            o_fh.write(pdb_content)
    return pdb_content


def random_sample_trajectory(traj, n_samples):
    """Samples a random subset of a trajectory."""
    random_ids = np.random.choice(traj.shape[0], n_samples,
                                  replace=traj.shape[0] < n_samples)
    return traj[random_ids]