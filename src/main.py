## This is pipeline code

"""
main.py - HDR → LDR pipeline
how to use: python main.py input.hdr output.png
"""
 
import sys
from src.generals.utils import load_hdr, save_ldr
from src.generals.preprocess import preprocess
from src.gradient_tone_mapping.tonemap import tonemap
from src.generals.postprocess import postprocess
 
 
def run(input_path: str, output_path: str):
    hdr = load_hdr(input_path)
    hdr = preprocess(hdr)
    ldr = tonemap(hdr)
    ldr = postprocess(ldr)
    save_ldr(ldr, output_path)
    print(f"save complete: {output_path}")
 
 
if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
 