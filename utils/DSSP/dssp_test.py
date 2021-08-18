from Bio.PDB import DSSP, PDBParser
import os

file='/data/proli/Bioinfo/utils/DSSP/5d7g.cif'

if __name__ == '__main__':
    p = PDBParser()
    structure = p.get_structure("5D7G", file)
    model = structure[0]
    dssp = DSSP(model, file)
    for row in dssp:
        with open("./sscasp11.dssp", "a") as q:
            #print(row[1：3])
            q.write(str(row[1:3]))
