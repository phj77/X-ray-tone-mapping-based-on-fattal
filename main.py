## This is pipeline code

"""
main.py - HDR → LDR pipeline
how to use: python main.py input.hdr output.png
"""
 
import sys
from utils import load_hdr, save_ldr
from preprocess import preprocess
from tonemap import tonemap
from postprocess import postprocess
 
 
def run(input_path: str, output_path: str):
    hdr = load_hdr(input_path)
    hdr = preprocess(hdr)
    ldr = tonemap(hdr)
    ldr = postprocess(ldr)
    save_ldr(ldr, output_path)
    print(f"save complete: {output_path}")
 
 
if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
 