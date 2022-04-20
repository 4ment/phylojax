import sys

import dendropy
from dendropy import DnaCharacterMatrix, Tree

from .tree import setup_dates, setup_indexes


def read_tree(
    tree_path, dated=True, heterochornous=True, taxa=dendropy.TaxonNamespace()
):
    tree_format = "newick"
    with open(tree_path) as fp:
        if next(fp).upper().startswith("#NEXUS"):
            tree_format = "nexus"

    tree = Tree.get(
        path=tree_path,
        schema=tree_format,
        tree_offset=0,
        taxon_namespace=taxa,
        preserve_underscores=True,
        rooting="force-rooted",
    )
    tree.resolve_polytomies(update_bipartitions=True)

    setup_indexes(tree)
    if dated:
        setup_dates(tree, heterochornous)
    return tree


def read_tree_and_alignment(
    tree_path, alignment, dated=True, heterochornous=True, **kwargs
):
    tree = read_tree(tree_path, dated, heterochornous)

    # alignment
    seqs_args = dict(schema="nexus", preserve_underscores=True)
    with open(alignment) as fp:
        if next(fp).startswith(">"):
            seqs_args = dict(schema="fasta")
    dna = DnaCharacterMatrix.get(
        path=alignment, taxon_namespace=tree.taxon_namespace, **seqs_args
    )
    sequence_count = len(dna)
    if sequence_count != len(dna.taxon_namespace):
        sys.stderr.write("taxon names in trees and alignment are different")
        exit(2)
    return tree, dna
