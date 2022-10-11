### IMPORTS AND STATIC PATHS ###
from bp3 import bepipred3
from pathlib import Path
import argparse
import sys

WORK_DIR = Path( Path(__file__).parent.resolve() )

### COMMAND LINE ARGUMENTS ###
parser = argparse.ArgumentParser("Make B-cell epitope predictions from fasta file.")
parser.add_argument("-i", required=True, action="store", dest="fasta_file", type=Path, help="Fasta file contianing antigens")
#parser.add_argument("-o", required=True, action="store", dest="out_dir", type=Path, help="Output file to store B-cell epitope predictions.")
#parser.add_argument("-pred", action="store", choices=["mjv_pred", "vt_pred"], required=True, dest="pred", help="Majorty vote ensemble prediction or\
#   variable threshold predicition on average ensemble posistive probabilities. ")
#parser.add_argument("-add_seq_len", action="store_true", dest="add_seq_len", help="Add sequence lengths to esm1b-encodings. Default is false.")
#parser.add_argument("-esm1b_dir", action="store", default= WORK_DIR / "esm1b_encodings", dest="esm1b_dir", type=Path, help="Directory to save ESM1b encodings to. Default is current working directory.")
parser.add_argument("-t", action="store", default=0.1512, type=float, dest="var_threshold", help="Threshold to use, when making predictions on average ensemble positive probability outputs. Default is 0.1512.")
parser.add_argument("-top", action="store", default=15, type=int, dest="top_cands", help="Number of top candidates to display in top candidate residue output file. Default is 15.")

args = parser.parse_args()
fasta_file = args.fasta_file
var_threshold = args.var_threshold
top_cands = args.top_cands

### Error handling ###

if var_threshold >= 1 or var_threshold <= 0:
    sys.exit(f"The threshold for B-cell epitope prediction must be a decimal between 0 and 1. Recevied value: {var_threshold}")

### CONSTANTS ###

add_seq_len = True
pred = "vt_pred"
esm1b_dir = WORK_DIR / "esm1b_encodings"
out_dir = WORK_DIR / "output_directory"
run_esm_model_local = WORK_DIR / "esm_model_state_dict.pt"

### MAIN ###

## Load antigen input and create ESM-1b encodings ## 
MyAntigens =  bepipred3.Antigens(fasta_file, esm1b_dir, add_seq_len=add_seq_len, run_esm_model_local=run_esm_model_local)
MyBP3EnsemblePredict = bepipred3.BP3EnsemblePredict(MyAntigens)
MyBP3EnsemblePredict.run_bp3_ensemble()
MyBP3EnsemblePredict.raw_ouput_and_top_epitope_candidates(out_dir, top_cands)
MyBP3EnsemblePredict.bp3_pred_variable_threshold(out_dir, var_threshold=var_threshold)

#generate plots (generating graphs for a maximum of 40 proteins)
MyBP3EnsemblePredict.bp3_generate_plots(out_dir, num_interactive_figs=30)