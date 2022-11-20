# BepiPred-3.0
BepiPred-3.0 uses the recent developments in protein language modeling to predict potential B-cell epitopes from protein sequence(s). 
## Input
* Upload a fasta formatted file containing the protein sequence(s).
* Set classification threshold (default is 0.1512).
* Specify x number of top residue candidates should be included for the protein sequence(s) (default is 15). 
* Use sequential smoothing (rolling mean) for B-cell epitope probability score graphs. This can make identification of epitope patches easier, but with a higher risk of false positives.
## Outputs
A total of 4 files are generated, where epitope and non-epitope residues are indicated with uppercase and lowercase letters respectively.  
* 'Bcell_epitope_preds.fasta'. Contains B-cell epitope predictions for the protein sequence(s) at the specified threshold is generated.
* 'Bcell_epitope_top_x_preds.fasta'. Contains the top x residue candidates. 
* 'raw_output.csv'. Contains the B-cell epitope probability scores for each residue of the protein sequence(s). The rolling mean score is also provided.
* 'output_interactive_figures.html'. The optimal threshold is often protein specific. This html file can be opened in any browser and allows the user to manually set the threshold for each protein and get the the corresponding B-cell epitope predictions. By default, these graphs are generated for the first 40 proteins in the fasta file due to file size contraints. 


## Graphical Output: Linear and discontinous B-cell epitope prediction
In the graphical output, B-cell epitope predictions are illustrated with bar plots. The threshold for predicting B-cell epitopes is often protein-specific, and single threshold is unlikey to be optimal for all proteins. We believe this intuitive interface allows researchers to maximize their precision of B-cell epitope prediction.

### Graph output without sequential smoothing (discontinous B-cell epitope prediction)
The x and y axis are protein sequence positions and BepiPred-3.0 epitope scores.
Residues with a higher score are more likely to be part of a B-cell epitope.
The threshold can be set by using the slider, which moves a dashed line along the y-axis.
Epitope predictions are updated according to the slider.
The B-cell epitope predictions at the set threshold can be downloaded by clicking the button 'Download epitope prediction'.

![Screenshot](GraphOutput.png)

### Graph output with sequential smoothing (linear B-cell epitope prediction)
If you chose to use the sequential smoothing (rolling mean) option, the graphical output will look different.
Using this option is more useful for detecting linear epitopes. But it is important to note, that some residues in the predicted linear epitope
are false positives, meaning that they do not interact directly with an antibody. This is because BepiPred-3.0 is trained on PDB crystal structures of ab-ag complexes, and to predict antigen residues that are in contact with an antibody (within 4 angstrom).
![Screenshot](GraphOutputWSeqSmooth.png)


## Reference
If you found BepiPred-3.0 useful in your research, please cite our paper: [BepiPred-3.0: Improved B-cell epitope prediction using protein language models](https://doi.org/10.1002/pro.4497)
