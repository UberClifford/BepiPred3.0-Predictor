# BepiPred-3.0
BepiPred-3.0 uses the recent developments in protein language modeling to predict potential B-cell epitopes from protein sequence(s). 
## Input
* Upload a fasta formatted file containing the protein sequence(s).
* Set classification threshold (default is 0.1512).
* Specify x number of top residue candidates should be included for the protein sequence(s) (default is 15). 

## Outputs
A total of 4 files are generated, where epitope and non-epitope residues are indicated with uppercase and lowercase letters respectively.  
* 'Bcell_epitope_preds.fasta'. Contains B-cell epitope predictions for the protein sequence(s) at the specified threshold is generated.
* 'Bcell_epitope_top_x_preds.fasta'. Contains the top x residue candidates. 
* 'raw_output.csv'. Contains the B-cell epitope probability scores for each residue of the protein sequence(s).
* 'output_interactive_figures.html'. The optimal threshold is often protein specific. This html file can be opened in any browser and allows the user to manually set the threshold for each protein and get the the corresponding B-cell epitope predictions. By default, these graphs are generated for the first 30 proteins in the fasta file due to file size contraints. 

## Plot information
![Screenshot](example_interactive_plot.png)	
In the interface for 'output_interactive_figures.html', the x and y axis are protein sequence positions and BepiPred-3.0 epitope scores. Residues with a higher score are more likely to be
part of a B-cell epitope. The threshold can be set by using the slider bar, which moves a dashed line along the y-axis. Epitope predictions are updated accordingly, and B-cell epitope predictions at the set threshold can be downloaded by clicking the button ‘Download epitope prediction’.

## Reference
If you found BepiPred-3.0 useful in your research, please cite this paper:


[BepiPred-3.0: Improved B-cell epitope prediction using protein language models](https://www.biorxiv.org/content/10.1101/2022.07.11.499418v1)