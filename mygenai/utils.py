# general utility functions (primarily for data exploration)import rdkit
import rdkit
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
def moltosvg(mol,molSize=(450,150),kekulize=True):
    # taken from
    # https://rdkit.blogspot.com/2015/02/new-drawing-code.html
    mc = rdkit.Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            rdkit.Chem.Kekulize(mc)
        except:
            mc = rdkit.Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')
