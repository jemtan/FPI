import os
import nibabel as nib
import numpy as np
import fpiSubmit

def predict_folder(input_dir,output_dir,mode,data):
    fpiSubmit.predict_folder(input_dir,output_dir,mode,data)
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

    parser.add_argument("-d","--data", type=str, help="can be either 'brain' or 'abdom'.", required=True)

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    mode = args.mode
    data = args.data

    predict_folder(input_dir, output_dir, mode, data)



