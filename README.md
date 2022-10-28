# BepiPred3.0-Predictor
BepiPred3.0 predicts B-cell epitopes from ESM-2 encodings of proteins sequences. You can you this github to setup for local installment of BepiPred-3.0.
We also provide BepiPred-3.0 is also provided as a [web service](https://services.healthtech.dtu.dk/service.php?BepiPred-3.0)
## Usage
### Set up
The dependencies for the BepiPred-3.0 are listed in requirements.txt. These dependencies can be installed with pip. We used the python version 3.8.8. Other python versions may or may not be compatable with the packages listed in requirements.txt. It is recommended that you install the dependencies in a virtual environment. 
```bash
$ pip3 install -r requirements.txt
```

### Usage
A commandline script for most general use cases is provided. It takes a fasta file as input and outputs a fasta file containing B-cell epitope predictions. Output looks something like this, (capitilization=predicted epitope residue)
```bash
>7lj4_B
...QQaQRELK..
```

The first run might take a while because the ESM-2 models need to be downloaded to your torch hub model cache, if they are not already there. 
Alternatively if you already have esm-2 models downloaded (esm2_t33_650M_UR50D.pt and esm2_t33_650M_UR50D-contact-regression.pt), you can use them directly (see bepipred3_CLI.py script)

### Inputs 

The required arguments are:

* -i fasta formatted file containing the protein sequence(s).
* -o Output directory to store B-cell epitope predictions.
* -pred {mjv_pred, vt_pred} Majorty vote ensemble prediction or variable threshold predicition on average ensemble posistive probabilities


Optional arguments are:


* -add_seq_len          Add sequence lengths to esm-encodings. Default is false.
* -esm_dir ESM_DIR      Directory to save esm encodings to. Default is current working directory.
* -t VAR_THRESHOLD      Threshold to use, when making predictions on average ensemble positive probability outputs. Default is 0.1512.
* -top TOP_CANDS        Number of top candidates to display in top candidate residue output file. Default is 10.
* -rolling_window_size  Window size to use for rolling average on B-cell epitope probability scores. Default is 9.
* -use_rolling_mean     Use rolling mean B-cell epitope probability score for plot. Default is false.

### Outputs
A total of 4 files are generated, where epitope and non-epitope residues are indicated with uppercase and lowercase letters respectively.
* 'Bcell_epitope_preds.fasta'. Contains B-cell epitope predictions for the protein sequence(s) at the specified threshold is generated.
* 'Bcell_epitope_top_x_preds.fasta'. Contains the top x residue candidates.
* 'raw_output.csv'. Contains the B-cell epitope probability scores for each residue of the protein sequence(s). The rolling mean score is also provided.
* 'output_interactive_figures.html'. The optimal threshold is often protein specific. This html file can be opened in any browser and allows the user to manually set the threshold for each protein and get the the corresponding B-cell epitope predictions. By default, these graphs are generated for the first 40 proteins in the fasta file due to file size contraints.

### Example

An example of a command from linux CLI,
python bepipred3_CLI.py -i example_antigens.fasta -o ./output/ -pred vt_pred 

This will ESM-2 encode sequences in example_antigens.fasta, make B-cell epitope predictions using the default threshold. Outputs files are stored in a directory clled 'output'.

For more info, you can run,
python bepipred3_CLI.py -h

### Graphical Output: Linear and discontinous B-cell epitope prediction


### Cite
If you found BepiPred-3.0 useful in your research, please cite,

[BepiPred-3.0: Improved B-cell epitope prediction using protein language models](https://www.biorxiv.org/content/10.1101/2022.07.11.499418v1)
