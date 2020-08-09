import dendropy
from dendropy import Tree, DnaCharacterMatrix
import sys
from .tree import setup_indexes, setup_dates


def read_tree_and_alignment(tree, alignment, dated=True, heterochornous=True, **kwargs):
    # tree
    taxa = dendropy.TaxonNamespace()
    tree_format = 'newick'
    with open(tree) as fp:
        if next(fp).upper().startswith('#NEXUS'):
            tree_format = 'nexus'

    tree = Tree.get(path=tree, schema=tree_format, tree_offset=0, taxon_namespace=taxa, preserve_underscores=True,
                    rooting='force-rooted')
    tree.resolve_polytomies(update_bipartitions=True)

    setup_indexes(tree)
    if dated:
        setup_dates(tree, heterochornous)


    # alignment
    seqs_args = dict(schema='nexus', preserve_underscores=True)
    with open(alignment) as fp:
        if next(fp).startswith('>'):
            seqs_args = dict(schema='fasta')
    dna = DnaCharacterMatrix.get(path=alignment, taxon_namespace=taxa, **seqs_args)
    alignment_length = dna.sequence_size
    sequence_count = len(dna)
    if sequence_count != len(dna.taxon_namespace):
        sys.stderr.write('taxon names in trees and alignment are different')
        exit(2)
    return tree, dna
