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
parser.add_argument("-add_seq_len", action="store_true", dest="add_seq_len", help="Add sequence lengths to esm-encodings. Default is false. This option is used for the web server.")
parser.add_argument("-esm_dir", action="store", default= WORK_DIR / "esm_encodings", dest="esm_dir", type=Path, help="Directory to save esm encodings to. Default is current working directory.")
parser.add_argument("-t", action="store", default=0.1512, type=float, dest="var_threshold", help="Threshold to use, when making predictions on average ensemble positive probability outputs. Default is 0.15.")
parser.add_argument("-top", action="store", default=0.3, type=float, dest="top_cands", help="Top percentage of epitope residues Default is top 30 pct.")
parser.add_argument("-rolling_window_size", default=9, type=int, dest="rolling_window_size", help="Window size to use for rolling average on B-cell epitope probability scores. Default is 9.")
#*for biolib*
#parser.add_argument("-use_rolling_mean", action="store_true", dest="use_rolling_mean", help="Use rolling mean B-cell epitope probability score for plot. Default is false.")
parser.add_argument("-use_rolling_mean", action="store", choices = ["Yes", "No"], default="No", dest="use_rolling_mean", help="Use rolling mean B-cell epitope probability score for plot. Default is false.")


args = parser.parse_args()
fasta_file = args.fasta_file
out_dir = args.out_dir
var_threshold = args.var_threshold
pred = args.pred
add_seq_len = args.add_seq_len
esm_dir = args.esm_dir
top_cands = args.top_cands
rolling_window_size = args.rolling_window_size

#*For biolib*
if args.use_rolling_mean == "Yes":
	use_rolling_mean = True
else:
    use_rolling_mean = False

### MAIN ###

## Load antigen input and create ESM-2 encodings ## 

#on webservices, we have the esm2 model stored locally. To work you need both esm2_t33_650M_UR50D.pt and the esm2_t33_650M_UR50D-contact-regression.pt stored in same directory
MyAntigens = bepipred3.Antigens(fasta_file, esm_dir, add_seq_len=add_seq_len, run_esm_model_local=WORK_DIR / "models" / "esm2_t33_650M_UR50D.pt")

#MyAntigens = bepipred3.Antigens(fasta_file, esm_dir, add_seq_len=add_seq_len)
MyBP3EnsemblePredict = bepipred3.BP3EnsemblePredict(MyAntigens, rolling_window_size=rolling_window_size)
MyBP3EnsemblePredict.run_bp3_ensemble()
MyBP3EnsemblePredict.raw_ouput_and_top_epitope_candidates(out_dir, top_cands)

## B-cell epitope predictions ##
if pred == "mjv_pred":
    MyBP3EnsemblePredict.bp3_pred_majority_vote(out_dir)
elif pred == "vt_pred":
    MyBP3EnsemblePredict.bp3_pred_variable_threshold(out_dir, var_threshold=var_threshold)

#generate plots (generating graphs for a maximum of 40 proteins)
MyBP3EnsemblePredict.bp3_generate_plots(out_dir, num_interactive_figs=40, use_rolling_mean=use_rolling_mean)
print("Click 'Download' to get the result files")
