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