import os
import nibabel as nib
import numpy as np
import fpiSubmit
import evalresults

def eval_folder(output_dir,label_dir,mode,data):
    evalresults.eval_dir(output_dir,label_dir,mode=mode,save_file=os.path.join(output_dir,data+"_"+mode+"_score.txt"))
    return


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-mode", type=str, default="pixel", help="can be either 'pixel' or 'sample'.", required=False)

    parser.add_argument("-d","--data", type=str, help="can be either 'brain' or 'abdom'.", required=True)

    args = parser.parse_args()

    label_dir = args.label
    output_dir = args.output
    mode = args.mode
    data = args.data

    eval_folder(output_dir, label_dir, mode, data)



