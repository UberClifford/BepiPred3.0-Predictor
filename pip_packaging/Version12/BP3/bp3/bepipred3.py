### IMPORTS ###
import math
import subprocess
import numpy as np
import torch
import esm
import csv
import torch.nn as nn
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

### STATIC PATHS ###
ROOT_DIR = Path( Path(__file__).parent.resolve() )
MODELS_PATH = ROOT_DIR / "BP3Models"
#ESM_SCRIPT_PATH = ROOT_DIR / "extract.py"

### SET GPU OR CPU ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU device detected: {device}")
else:
    device = torch.device("cpu")
    print(f"GPU device not detected. Using CPU: {device}")

### MODEL ###

class MyDenseNetWithSeqLen(nn.Module):
    def __init__(self,
                 esm_embedding_size = 1281,
                 fc1_size = 150,
                 fc2_size = 120,
                 fc3_size = 45,
                 fc1_dropout = 0.7,
                 fc2_dropout = 0.7,
                 fc3_dropout = 0.7,
                 num_of_classes = 2):
        super(MyDenseNetWithSeqLen, self).__init__()
        
        
        self.esm_embedding_size = esm_embedding_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout
        
        self.ff_model = nn.Sequential(nn.Linear(esm_embedding_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, num_of_classes))
    
    def forward(self, antigen):
        batch_size = antigen.size(0)
        seq_len = antigen.size(1)
        #convert dim (N, L, esm_embedding) --> (N*L, esm_embedding)
        output = torch.reshape(antigen, (batch_size*seq_len, self.esm_embedding_size))
        output = self.ff_model(output)                                               
        return output

class MyDenseNet(nn.Module):
    def __init__(self,
                 esm_embedding_size = 1280,
                 fc1_size = 150,
                 fc2_size = 120,
                 fc3_size = 45,
                 fc1_dropout = 0.7,
                 fc2_dropout = 0.7,
                 fc3_dropout = 0.7,
                 num_of_classes = 2):
        super(MyDenseNet, self).__init__()
        
        
        self.esm_embedding_size = esm_embedding_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.fc1_dropout = fc1_dropout
        self.fc2_dropout = fc2_dropout
        self.fc3_dropout = fc3_dropout
        
        self.ff_model = nn.Sequential(nn.Linear(esm_embedding_size, fc1_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc1_dropout),
                                      nn.Linear(fc1_size, fc2_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc2_dropout),
                                      nn.Linear(fc2_size, fc3_size),
                                      nn.ReLU(),
                                      nn.Dropout(fc3_dropout),
                                      nn.Linear(fc3_size, num_of_classes))
    
    def forward(self, antigen):
        batch_size = antigen.size(0)
        seq_len = antigen.size(1)
        #convert dim (N, L, esm_embedding) --> (N*L, esm_embedding)
        output = torch.reshape(antigen, (batch_size*seq_len, self.esm_embedding_size))
        output = self.ff_model(output)                                               
        return output

### CLASSES ###

class Antigens():
    def __init__(self, fasta_file, esm_encoding_dir,
        add_seq_len=False, run_esm_model_local=None):
        """
        Initialize Antigens class object
        Inputs:
            device: pytorch device to use, default is cuda if available else cpu.
        """

        self.esm_encoding_dir = esm_encoding_dir
        self.run_esm_model_local = run_esm_model_local

        try:
            esm_encoding_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory for esm encodings already there. Saving encodings there.")
        else:
            print("Directory for esm encodings not found. Made new one.")

        self.accs, self.seqs = self.read_accs_and_sequences_from_fasta(fasta_file)
        num_of_seqs = len(self.seqs)
        #self.create_fasta_for_esm_transformer()
        self.add_seq_len = add_seq_len
        print(f"Number of sequences detected in fasta file: {num_of_seqs}")
        print(f"ESM-2 encoding sequences. Saving encodings to {str(esm_encoding_dir)}")
        self.esm_encoding_paths = self.get_esm2_represention_on_accs_seqs()
        self.ensemble_preds = None
        self.ensemble_probs = None

    def check_accepted_AAs(self, accs, sequences):
        accepted_AAs = set(["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"])
        entries = list( zip(accs, sequences) ) 
    
        for entry in entries:
            acc = entry[0]
            seq = entry[1]
            #if not accessions or sequences obtained (empty)
            check = all(res.upper() in accepted_AAs for res in seq)
            if not check:
                sys.exit(f"Nonstandard amino acid character detected in acc: {acc}. Allowed character lower and uppercase amino acids:\n{accepted_AAs}")

    def tuple_generator(self, data, batch_size=10):
        length = len(data)
        for i in range(0, length, batch_size):
            yield data[i:i + batch_size]


    def get_esm2_represention_on_accs_seqs(self):
        """
        data: list of tuples: [(seq_name, sequence)...]
        per_res_representations:
        """

        # Load ESM-2 model
        if self.run_esm_model_local is not None:
            print(f"Loading ESM2 from: {self.run_esm_model_local}")
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.run_esm_model_local)
            
        else: model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        #esm_representations = []
        #preparing batch for ESM2
        upper_case_sequences = [s.upper() for s in self.seqs]
        data = list(zip(self.accs, upper_case_sequences))
        nr_seqs = len(data)
        batch_generator = self.tuple_generator(data)
        
        enc_id = 0
        enc_paths = []

        for b in batch_generator:

            batch_labels, batch_strs, batch_tokens = batch_converter(b)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            acc_names = batch_labels

            # Extract per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]
    
            #encoding sequences
            for i, tokens_len in enumerate(batch_lens):

                esm_representation = token_representations[i, 1 : tokens_len - 1]
                if self.add_seq_len:
                    esm_representation = self.add_seq_len_feature(esm_representation)

                enc_path = self.esm_encoding_dir / f"{acc_names[i]}_{enc_id}.pt"
                torch.save(esm_representation, enc_path)

                enc_paths.append(enc_path)
                enc_id += 1

                print(f"ESM-2 encoded sequence {acc_names[i]} {enc_id}/{nr_seqs}")

        return enc_paths
    
    def read_accs_and_sequences_from_fasta(self, infile):
        """
        Input: readfile: Fasta file. 
        Outputs: List of tuples. Containing accs and sequences, e.g. [(acc, aTHNtem..)..()]. 
        """
        
        if not infile.is_file():
            sys.exit(f"The input file was invalid: {infile}")

        accs = list()
        sequences = list()
        seq = ""

        read_acc = False    
        infile = open(infile, "r")
        readfile = infile.readlines()
        
        infile.close()

        for line in readfile:
            line = line.strip()
            if line.startswith(">"):
                acc = line.split(">")[1]
                if read_acc:
                    accs.append(acc)
                    sequences.append(seq)
                    #reset sequence string
                    seq = ""
                #catch first accesion.
                else:
                    accs.append(acc)
            else:
                seq += line
                read_acc = True

        #get last sequence
        sequences.append(seq)

        if accs == False or sequences == False:
            sys.exit(f"No accessions or sequences found in fasta file. Please check file: {infile}")

        #check if there are characters that are not accepted by the ESM-2 transformer.
        self.check_accepted_AAs(accs, sequences)

        return accs, sequences
                
    def add_seq_len_feature(self, X):
        #adding sequence length to each positional ESM-2 embedding
        seq_len = X.size()[0]
        seq_len_v = torch.ones(seq_len)*seq_len
        seq_len_v = seq_len_v.unsqueeze(dim=1)
        new_X = torch.cat((X, seq_len_v), axis=1)
        
        return new_X
   
class BP3EnsemblePredict():
    
    def __init__(self,
                 antigens,
                 device = None,
                 rolling_window_size = 9,
                 top_pred_pct=0.3):
        """
        Inputs and initialization:
            antigens: Antigens class object
            device: pytorch device to use, default is cuda if available else cpu.
            
        """
        
        self.bp3_ensemble_run = False
        self.antigens = antigens
        self.rolling_window_size = rolling_window_size 
        self.top_pred_pct = top_pred_pct

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        if antigens.add_seq_len:
            self.model_architecture = MyDenseNetWithSeqLen()
            m_path = MODELS_PATH / "BP3C50IDSeqLenFFNN" 
            self.model_states = list( m_path.glob("*Fold*") )
            self.classification_thresholds = {'Fold1': 0.21326530612244898,
                                              'Fold2': 0.15510204081632653,
                                              'Fold3': 0.1163265306122449,
                                              'Fold4': 0.09693877551020408,
                                              'Fold5': 0.19387755102040816}
            self.threshold_keys = [model_state.stem for model_state in self.model_states] 

        else:
            self.model_architecture = MyDenseNet()
            m_path = MODELS_PATH / "BP3C50IDFFNN" 
            self.model_states = list( m_path.glob("*Fold*") )
            self.classification_thresholds = {'Fold1': 0.2326530612244898,
                                              'Fold2': 0.15510204081632653,
                                              'Fold3': 0.1163265306122449,
                                              'Fold4': 0.15510204081632653,
                                              'Fold5': 0.19387755102040816}
            self.threshold_keys = [model_state.stem for model_state in self.model_states]


        #user specified classification thresholds for each fold
#        if classification_thresholds != None:
#            self.classification_thresholds = classification_thresholds


    def compute_rolling_mean_on_bp3_prob_outputs(self, antigen_avg_ensemble_probs):
        #same=ensures that the rolling mean output will have the same length as the number of residues for antigen.
        antigen_avg_ensemble_probs = antigen_avg_ensemble_probs.cpu().detach().numpy()
        return np.convolve(antigen_avg_ensemble_probs, np.ones(self.rolling_window_size), 'same') / self.rolling_window_size

    def run_bp3_ensemble(self):
        """
        INPUTS: antigens: Antigens() class object.  
        
        OUTPUTS:
                No outputs. Stores probabilities of ensemble models in Antigens() class object.
                Run bp3_pred_variable_threshold() or bp3_pred_majority_vote() afterwards to make predictions. 
        """
        num_of_models = len(self.model_states)
        ensemble_probs = list()
        threshold_keys = list()
        softmax_function = nn.Softmax(dim=1)
        model = self.model_architecture

        data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.esm_encoding_paths) )
        print("Generating BepiPred-3.0 scores")
        for acc, seq, esm_encoding_path in data:
            ensemble_prob = list()
            all_model_preds = list()
            num_residues = len(seq)
            esm_encoding = torch.load(esm_encoding_path)
            esm_encoding = torch.unsqueeze(esm_encoding, 0).to(self.device)
            
            for i in range(num_of_models):
                with torch.no_grad():
                
                    model_state = self.model_states[i] 
                    model.load_state_dict(torch.load(model_state, map_location=self.device))
                    model = model.to(self.device)
                    model.eval()
                    model_output = model(esm_encoding)
                    model_probs = softmax_function(model_output)[:, 1]

                    ensemble_prob.append(model_probs)

            ensemble_probs.append(ensemble_prob)
        
        self.bp3_ensemble_run = True
        self.antigens.ensemble_probs = ensemble_probs


    def create_toppct_files(self, outfile_path):
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory B-cell epitope predictions already there. Saving results there.")

        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
                Use method run_bp3_ensemble(antigens).")
        else:
            antigens = self.antigens
            data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.ensemble_probs) )
            epi_toppct_outfile_content = str()
            linepi_toppct_outfile_content = str()

            for acc, seq, ensemble_prob in data:
                num_residues  = len(seq)
                avg_prob = torch.mean(torch.stack(ensemble_prob, axis=1), axis=1)
                avg_prob_rolling_mean = self.compute_rolling_mean_on_bp3_prob_outputs(avg_prob) 
                ensemble_pred_len = len(avg_prob)

                if ensemble_pred_len < num_residues:
                    print(f"Sequence longer than what the ESM-2 trasnformer can encode entirely, {acc}. Outputting predictions up till {ensemble_pred_len} position.")

                #get top % predictions
                nr_top_cands = round(num_residues*self.top_pred_pct)
                
                top_preds =[res_idx for res_idx, _ in sorted( [(idx, p) for idx, p in enumerate(avg_prob)], key=lambda pair: pair[1],  reverse=True)][:nr_top_cands]
                epitope_preds = "".join([seq[i].upper() if i in top_preds else seq[i].lower() for i in range(ensemble_pred_len)])
                epi_toppct_outfile_content += f">{acc}\n{epitope_preds}\n"

                top_preds =[res_idx for res_idx, _ in sorted( [(idx, p) for idx, p in enumerate(avg_prob_rolling_mean)], key=lambda pair: pair[1],  reverse=True)][:nr_top_cands]
                epitope_preds = "".join([seq[i].upper() if i in top_preds else seq[i].lower() for i in range(ensemble_pred_len)])
                linepi_toppct_outfile_content += f">{acc}\n{epitope_preds}\n"


            #write top candidates file
            outfile_content = epi_toppct_outfile_content[:-1]
            with open(outfile_path  / f"Bcell_epitope_top_{round(self.top_pred_pct*100)}pct_preds.fasta", "w") as outfile: outfile.write(outfile_content)
            outfile_content = linepi_toppct_outfile_content[:-1]
            with open(outfile_path  / f"Bcell_linepitope_top_{round(self.top_pred_pct*100)}pct_preds.fasta", "w") as outfile: outfile.write(outfile_content)

    def create_csvfile(self, outfile_path):
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory B-cell epitope predictions already there. Saving results there.")

        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
                Use method run_bp3_ensemble(antigens).")
        else:
            antigens = self.antigens
            data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.ensemble_probs) )
            combined_csv_format = str()
            combined_csv_format += "Accession,Residue,BepiPred-3.0 score,BepiPred-3.0 linear epitope score"

            for acc, seq, ensemble_prob in data:
                num_residues  = len(seq)
                avg_prob = torch.mean(torch.stack(ensemble_prob, axis=1), axis=1)
                #avg_prob = np.mean(np.stack(ensemble_prob, axis=1), axis=1)
                avg_prob_rolling_mean = self.compute_rolling_mean_on_bp3_prob_outputs(avg_prob) 
                ensemble_pred_len = len(avg_prob)

                if ensemble_pred_len < num_residues:
                    print(f"Sequence longer than what the ESM-2 trasnformer can encode entirely, {acc}. Outputting predictions up till {ensemble_pred_len} position.")

                csv_format = [f"{acc},{seq[i].upper()},{avg_prob[i]}, {avg_prob_rolling_mean[i]}" for i in range(ensemble_pred_len)]
                csv_format = "\n".join(csv_format)
                combined_csv_format += f"\n{csv_format}"

            #write raw csv file
            with open(outfile_path  / "raw_output.csv", "w") as outfile:
                outfile.write(combined_csv_format)

        
    def bp3_pred_variable_threshold(self, outfile_path, var_threshold = 0.1512):
        
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory B-cell epitope predictions already there. Saving results there.")
        else:
            print("Directory B-cell epitope predictions not found. Made new one. ")

        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
 Use method run_bp3_ensemble(antigens).")
        else:
            data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.ensemble_probs) )
            ensemble_preds = list()
            outfile_content = str()
            
            #go through each antigen
            for acc, seq, ensemble_prob in data:
                all_model_preds = list()
                num_residues = len(seq)
                #avg_prob = np.mean(np.stack(ensemble_prob, axis=1), axis=1)
                avg_prob = torch.mean(torch.stack(ensemble_prob, axis=1), axis=1)
                ensemble_pred = [1 if res >= var_threshold else 0 for res in avg_prob]

                ensemble_pred_len = len(ensemble_pred)
                if ensemble_pred_len < num_residues:
                    print(f"Sequence longer than what the ESM-2 trasnformer can encode entirely, {acc}. Outputting predictions up till {ensemble_pred_len} position.")

                epitope_preds = "".join([seq[i].upper() if ensemble_pred[i] == 1 else seq[i].lower() for i in range( ensemble_pred_len )])
                outfile_content += f">{acc}\n{epitope_preds}\n"
                ensemble_preds.append(ensemble_pred)
            
            self.antigens.ensemble_preds = ensemble_preds
            outfile_content = outfile_content[:-1]
            #saving output to fasta formatted output file
            with open(outfile_path  / "Bcell_epitope_preds.fasta", "w") as outfile:
                outfile.write(outfile_content)
            
    def bp3_pred_majority_vote(self, outfile_path):
        """
        
        """
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory B-cell epitope predictions already there. Saving results there.")
        else:
            print("Directory B-cell epitope predictions not found. Made new one. ")
        
        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
 Use method run_bp3_ensemble(antigens).")
        else:
            data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.ensemble_probs) )
            ensemble_preds = list()
            outfile_content = str()
            
            #go through each antigen
            for acc, seq, ensemble_prob in data:
                all_model_preds = list()
                num_residues = len(seq)
                
                #collect all predictions of all models in ensemble
                for i in range( len(ensemble_prob) ):
                    model_probs = ensemble_prob[i]
                    classification_threshold = self.classification_thresholds[ self.threshold_keys[i] ]
                    model_preds = [1 if res >= classification_threshold else 0 for res in model_probs]
                    all_model_preds.append(model_preds)
                    
                #ensemble majority vote 
                ensemble_pred = np.asarray(all_model_preds)
                ensemble_pred_len = np.shape(ensemble_pred)[1]

                if ensemble_pred_len < num_residues:
                    print(f"Sequence longer than what the ESM-2 trasnformer can encode entirely, {acc}. Outputting predictions up till {ensemble_pred_len} position.")
                
                ensemble_pred = [np.argmax( np.bincount(ensemble_pred[:, i]) ) for i in range(ensemble_pred_len)]
                epitope_preds = "".join([seq[i].upper() if ensemble_pred[i] == 1 else seq[i].lower() for i in range( len(ensemble_pred) )])
                outfile_content += f">{acc}\n{epitope_preds}\n"
                ensemble_preds.append(ensemble_pred)
            
            self.antigens.ensemble_preds = ensemble_preds
            outfile_content = outfile_content[:-1]
            #saving output to fasta formatted output file
            with open(outfile_path / "Bcell_epitope_preds.fasta", "w") as outfile:
                outfile.write(outfile_content)


    def add_line_breaks(self, seq, every_x_line = 128):
        seq_len = len(seq)
        line_breaks = seq_len / every_x_line
    
        if line_breaks < 1:
            parsed_seq = seq
        else:
            num_line_breaks = int(math.floor(line_breaks))        
            line_break_at_chars = [every_x_line*i for i in range(1, num_line_breaks + 1)]
    
            for i in range(len(line_break_at_chars)):
                #+4 because <br> is 4 chars
                seq = seq[:line_break_at_chars[i]+i*4]+"<br>"+seq[line_break_at_chars[i]+i*4:]
    
            parsed_seq = seq
    
        return parsed_seq
    
    def filter_thresholds(self, residues, thresholds, ensemble_pred_len, avg_prob):
        """
        function for filtering out thresholds, which dont change the prediction. 
        This is an optimization, so that a lot fewer frames are used per plotly plot.
        """
        nr_thresholds = len(thresholds)
        prev_pred = "".join([residues[i].upper() if avg_prob[i] >= thresholds[0] else residues[i].lower() for i in range(ensemble_pred_len)])
        filtered_thresholds = list()
        filtered_thresholds.append(thresholds[0])
        for i in range(1, nr_thresholds):
            t = thresholds[i]
            current_pred = "".join([residues[i].upper() if avg_prob[i] >= t else residues[i].lower() for i in range(ensemble_pred_len)])
            if current_pred != prev_pred:
                filtered_thresholds.append(t)
                prev_pred = current_pred
            else:
                pass

        filtered_thresholds = np.asarray(filtered_thresholds)
    
        return filtered_thresholds
    
    def insert_into_html(self, pattern, html_file_content, what_to_insert):
        """
        For inserting java code into html file 
        """
        
        search_from = 0
        idx_to_insert_at = 0
        while idx_to_insert_at != -1:
    
            idx_to_insert_at = html_file_content.find(pattern, search_from + 1)
            if idx_to_insert_at != -1:
                html_file_content = html_file_content[:idx_to_insert_at] + what_to_insert + html_file_content[idx_to_insert_at:]
                search_from = idx_to_insert_at+len(what_to_insert)
    
        return html_file_content
    
    
    def bp3_generate_plots(self, outfile_path, var_threshold=0.1512, num_interactive_figs=50, use_rolling_mean=False):
        try:
            outfile_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Directory for saving plots already there. Saving results there.")
        else:
            print("Directory for saving plots not found. Made new one. ")
        
        if not self.bp3_ensemble_run:
            sys.exit("BP3 ensemble has not been run, so predictions cannot be made.\
 Use method run_bp3_ensemble(antigens).")
        else:
            #thresholds to initially test
            nr_thresholds = 400
            #line split on text box
            every_x_line = 128
            thresholds = np.round(np.linspace(0, 1, nr_thresholds), decimals=5)
            y_init = [var_threshold, var_threshold]
            interactive_figure_list = list()
            data = list( zip(self.antigens.accs, self.antigens.seqs, self.antigens.ensemble_probs) )

            print("Creating figures")
            for acc, seq, ensemble_prob in data:

                epitope_preds_at_diff_thresh = list()
                seq_len  = len(seq)
                avg_prob = torch.mean(torch.stack(ensemble_prob, axis=1), axis=1)
                
                if use_rolling_mean and seq_len >= self.rolling_window_size:
                    avg_prob = self.compute_rolling_mean_on_bp3_prob_outputs(avg_prob)
                    plot_title = f"BepiPred-3.0 linear epitope scores on {self.add_line_breaks(acc)}"

                else:
                    avg_prob = avg_prob.cpu().detach().numpy()
                    plot_title = f"BepiPred-3.0 epitope scores on {self.add_line_breaks(acc)}"
                    
                    if use_rolling_mean:
                        plot_title += ".<br>Sequence was too short for linear epitope scoring."


                #compute rolling mean here
                ensemble_pred_len = len(avg_prob)

                if ensemble_pred_len < seq_len:
                    plot_title += f".<br>Sequence too long. Predictions are truncated to the first {ensemble_pred_len} residues."                    
                
                res_counts = [i for i in range(1, ensemble_pred_len + 1)]
                residues =  [seq[i] for i in range(ensemble_pred_len)]
                acc_col = [acc for i in range(ensemble_pred_len)]
            
                #quick solution for for setting some figure update height according sequence length (number of line breaks in text box)
                line_breaks = ensemble_pred_len  / every_x_line
                line_breaks_in_acc = len(acc) / every_x_line
                num_line_breaks = int(math.floor(line_breaks)) + int(math.floor(line_breaks_in_acc))
                if num_line_breaks <= 1:
                    figure_height_update = 200
                    y_coord = -0.37
                else:
                    figure_height_update = 200 + num_line_breaks*8
                    y_coord = -0.37 - 0.05*num_line_breaks
            
                d = {"BP3EpiProbScore": avg_prob, "Residue": residues, "SeqPos": res_counts, "Accession": acc_col}
                df = pd.DataFrame(data=d)
                df.round({"BP3EpiProbScore":5})
            
                # filter only useful thresholds (where prediction changes)
                filtered_thresholds = self.filter_thresholds(residues, thresholds, ensemble_pred_len, avg_prob)

                #insert default/user specified threshold
                indices_above = np.argwhere(filtered_thresholds >= var_threshold)

                if indices_above.size > 0:
                    first_indice_above = indices_above[0][0]
                    filtered_thresholds = np.insert(filtered_thresholds, first_indice_above, var_threshold)
                    
                else:
                    first_indice_above = -1
                    np.append(filtered_thresholds, var_threshold)
                
                for t in filtered_thresholds:
                    epitope_preds_at_diff_thresh.append(f"{self.add_line_breaks(acc)} (A/a=Epitope/Non-epitope), Threshold: {t}<br>"+self.add_line_breaks("".join([residues[i].upper() if avg_prob[i] >= t else residues[i].lower() for i in range(ensemble_pred_len )])))
                
                
                #create initial figure
                x_data = [1, ensemble_pred_len]
                fig = go.Figure()
                bar_fig = px.bar(df, x="SeqPos", y="BP3EpiProbScore", hover_data=["Residue", "Accession"], title= plot_title, color="BP3EpiProbScore", labels={'BP3EpiProbScore':'BepiPred-3.0 epitope score', "SeqPos": "Sequence position"}, height=700)
                stored_coloraxis_info = bar_fig.layout.coloraxis
                fig.add_trace(bar_fig.data[0])
                fig.add_trace(go.Scatter(x=x_data, y=y_init, mode="lines", line_color="#2C596A", line={"dash":"dash"}, showlegend=False))
                fig.add_annotation(text=f"<b>{epitope_preds_at_diff_thresh[first_indice_above]}</b>", align='left', showarrow=False, xref='paper', yref='paper', x=0.0, y=y_coord, bgcolor="#F9C68F", font_family='Courier New, monospace', bordercolor = "#2C596A")
                fig.update_layout(margin=dict(b=figure_height_update), title=plot_title, xaxis_title="Sequence position", yaxis_title="BepiPred-3.0 epitope score", coloraxis = stored_coloraxis_info, height=700)
                
                #only create x number of interactive figures due file size constraints.
                thresholds_to_plot = [[t,t] for t in filtered_thresholds]
                #thresholds_to_plot = thresholds_to_plot[:first_indice_above] + [[var_threshold, var_threshold]] + thresholds_to_plot[first_indice_above:]
                if len(interactive_figure_list) < num_interactive_figs:
                    #get frames for threshold animated figure
                    frames = []
                    for i in range(len(filtered_thresholds)):
                        layout_i = go.Layout(annotations=[go.layout.Annotation(text=f"<b>{epitope_preds_at_diff_thresh[i]}</b>", align='left', showarrow=False, xref='paper', yref='paper', x=0.0, y=y_coord, bgcolor="#F9C68F", font_family='Courier New, monospace', bordercolor = "#2C596A")])
                        frames.append(go.Frame(name=str(i), data=[go.Scatter(x=x_data, y=thresholds_to_plot[i], mode="lines", line_color="#2C596A", line={"dash":"dash"}, showlegend=False), bar_fig.update_traces().data[0],], layout=layout_i))
                    
                    #update figure with all the frames
                    figa = go.Figure(data=fig.data, frames=frames, layout=fig.layout)
                    #create threshold sliders
                    figa.update_layout(height=700, title=plot_title, xaxis_title="Sequence position", yaxis_title="BepiPred-3.0 epitope score", coloraxis = stored_coloraxis_info)
                    figa.update_layout(
                        sliders=[
                            {
                                "active": 0,
                                "currentvalue": {"xanchor":"left", "prefix": "<b>Threshold: </b>", "font": {"color":"#2C596A"}},
                                "len": 0.9,
                                "tickcolor": "#2C596A",
                                "font": {"color": "#2C596A"},
                                "steps": [      
                                    {
                                        "args": [
                                            [figa.frames[j].name],
                                            {
                                                "frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",
                                                "fromcurrent": True,
                                                "font": {"color":"white"}
                                            },
                                        ],
                                        "label": filtered_thresholds[j],
                                        "method": "animate",
                                    }
                                    for j in range(len(figa.frames))
                                ],
                            }
                        ],
                    )
                    interactive_figure_list.append(figa)
            
            #write interactive plots
            with open(outfile_path / "output_interactive_figures.html", 'w') as f:
                for figure in interactive_figure_list:
                    f.write(figure.to_html(full_html=False, include_plotlyjs='cdn', auto_play=False))
            
            ## Manually insert some custom java code ##
            
            #download B-cell epitope predictions button button
            #insert line before Plotly.newPlot
            button_pattern = b"""Plotly.newPlot"""
            plotly_java_download_button = b"""var config = {
              modeBarButtonsToAdd: [
                {
                  name: 'Download epitope predictions',
                  icon: Plotly.Icons.disk,
                  direction: 'up',
                  click: function(gd){let str = new String(gd.layout.annotations[0].text.toString()); let filename = new String(gd.layout.title.text.toString()).replaceAll(" ", "_"); let seq = ">" + str.replaceAll("<b>", "").replaceAll("</b>", "").replaceAll("<br>", "\\n"); let blob = new Blob([seq], {type: "text/plain"}); let link = document.createElement("a"); let url = URL.createObjectURL(blob); link.setAttribute("href", url); link.setAttribute("download", filename + ".fasta"); link.style.visibility = 'hidden'; document.body.appendChild(link); link.click(); document.body.removeChild(link);
                }}],}\n"""
            
            #set download button in config 
            #insert before {"responsive": true}  
            config_pattern = b"""{"responsive": true}"""
            plotly_config = b"""config, """
            
            ## edit interactive html ##
            
            with open(outfile_path / "output_interactive_figures.html", "rb") as infile:
                html_file_content = infile.read()
            
            #insert java code
            new_html_content = self.insert_into_html(button_pattern, html_file_content, plotly_java_download_button)
            new_html_content = self.insert_into_html(config_pattern, new_html_content, plotly_config)
            new_html_content = new_html_content.decode("utf-8")
            with open(outfile_path / "output_interactive_figures.html", "w") as outfile:
                outfile.write(new_html_content)



### OLD ###

#   def call_esm_script(self):
#       fastaPath = self.esm_encoding_dir / "antigens.fasta"
#       
#       try: #only using this for biolib implementation 
#           if self.run_esm_model_local is not None:
#               if self.run_esm_model_local.is_file():
#                   subprocess.check_call(['python', ESM_SCRIPT_PATH, self.run_esm_model_local, fastaPath, self.esm_encoding_dir, "--include", "per_tok"])
#               else:
#                   sys.exit(f"Could not find local model: {self.run_esm_model_local}.")
#           else:
#               subprocess.check_call(['python', ESM_SCRIPT_PATH, "esm2_t33_650M_UR50D", fastaPath, self.esm_encoding_dir, "--include", "per_tok"])
#       except subprocess.CalledProcessError as error:
#           sys.exit(f"ESM model could not be run with following error message: {error}.\nThis is likely a memory issue.")


#   def prepare_esm_data(self):
#       
#       esm_representations = list()
#       for acc in self.accs:
#           esm_encoded_acc = torch.load(self.esm_encoding_dir / f"{acc}.pt")
#           esm_representation = esm_encoded_acc["representations"][33]
#           
#           if self.add_seq_len:
#               esm_representation = self.add_seq_len_feature(esm_representation)
#           
#           esm_representations.append(esm_representation)
#       
#       return esm_representations



#    def create_fasta_for_esm_transformer(self):
#        """
#        Outputs fasta file accesions and sequences into a fasta file format, that can be read by ESM-2 transformer.  
#        """
#        uppercase_entries = list()
#        #convert all sequences to uppercase
#        entries = list( zip(self.accs, self.seqs) )
#
#        for entry in entries:
#            acc  = entry[0]
#            sequence = entry[1]
#            upper_case_sequence = sequence.upper()
#            uppercase_entries.append( (acc, upper_case_sequence) )
#
#
#        with open(self.esm_encoding_dir / "antigens.fasta", "w") as outfile:
#            output = str()
#            for entry in uppercase_entries :
#                output += f">{entry[0]}\n{entry[1]}\n"
#
#            output = output[:-1]
#            outfile.write(output)