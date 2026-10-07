"""Erase the wearer's own clothing where the new garment does not cover it.

    remover.py           ClothingRemover: clothing mask, fill, removal
    background.py        the room behind the wearer, learned from the feed; body silhouette
    parsing_schp.py      SCHP human parser (ATR + LIP, ONNX on CoreML), the live default
    parsing_fashn.py     FASHN parser, better but slower (offline)
"""
