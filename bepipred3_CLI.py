### IMPORTS AND STATIC PATHS ###
from bp3 import bepipred3
from pathlib import Path
import argparse

WORK_DIR = Path( Path(__file__).parent.resolve() )

### COMMAND LINE ARGUMENTS ###
parser = argparse.ArgumentParser("Make B-cell epitope predictions from fasta file.")
parser.add_argument("-i", required=True, action="store", dest="fasta_file", type=Path, help="Fasta file contianing antigens")
parser.add_argument("-o", required=True, action="store", dest="out_dir", type=Path, help="Output file to store B-cell epitope predictions.")
parser.add_argument("-pred", action="store", choices=["mjv_pred", "vt_pred"], required=True, dest="pred", help="Majorty vote ensemble prediction or\
	variable threshold predicition on average ensemble posistive probabilities. ")
parser.add_argument("-add_seq_len", action="store_true", dest="add_seq_len", help="Add sequence lengths to esm1b-encodings. Default is false.")
parser.add_argument("-esm1b_dir", action="store", default= WORK_DIR / "esm1b_encodings", dest="esm1b_dir", type=Path, help="Directory to save ESM1b encodings to. Default is current working directory.")
parser.add_argument("-t", action="store", default=0.15, type=float, dest="var_threshold", help="Threshold to use, when making predictions on average ensemble positive probability outputs. Default is 0.15.")
parser.add_argument("-top", action="store", default=10, type=int, dest="top_cands", help="Number of top candidates to display in top candidate residue output file. Default is 10.")

args = parser.parse_args()
fasta_file = args.fasta_file
out_dir = args.out_dir
var_threshold = args.var_threshold
pred = args.pred
add_seq_len = args.add_seq_len
esm1b_dir = args.esm1b_dir
top_cands = args.top_cands

### MAIN ###

## Load antigen input and create ESM-1b encodings ## 
MyAntigens =  bepipred3.Antigens(fasta_file, esm1b_dir, add_seq_len=add_seq_len)
MyBP3EnsemblePredict =  bepipred3.BP3EnsemblePredict(MyAntigens)
MyBP3EnsemblePredict.run_bp3_ensemble(MyAntigens)

MyBP3EnsemblePredict.raw_ouput_and_top_epitope_candidates(MyAntigens, out_dir, top_cands)

## B-cell epitope predictions ##
if pred == "mjv_pred":
    MyBP3EnsemblePredict.bp3_pred_majority_vote(MyAntigens, out_dir)
elif pred == "vt_pred":
    MyBP3EnsemblePredict.bp3_pred_variable_threshold(MyAntigens, out_dir, var_threshold=var_threshold)