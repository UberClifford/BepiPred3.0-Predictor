# BepiPred3.0-Predictor
BepiPred3.0 predicts B-cell epitopes from ESM1-b encodings of proteins sequences. 
The ESM-1b transformer is intergrated in this repository, so no create this separately. 
## Usage

### Setting up virtual environment
Clone repository from a Git CLI
```bash
$ git clone https://github.com/UberClifford/BepiPred3.0-Predictor.git
```
Reconstruct virtual anaconda environmnet from .yml on Windows OS or UNIX OS
```bash
$ conda env create -f WindowsOSEnvironment.yml
```
```bash
or
$ conda env create -f  UNIXOSEnvironment.yml
```
They are 5,89 and x GB in size respectively.  

NOTE: Comes with a pytorch installment of the CUDA 11.3 toolkit, which may not be compatible with your GPU.
If not, you need to install pytorch with the appropriate toolkit. The virtual environment does not come with a jupyter notebook installment and will only work bepipred3_CLI.py. So if you want this functionality you'll need to install it in the virtual  environment.

### Using commandline script 
A commandline script for most general use cases is provided. It takes a fasta file as input and outputs a fasta file containing B-cell epitope predictions. Output looks something like this,
```bash
>7lj4_B
...QQAQRELK..
...EE-EEEEE...
```
An example of a command from bash command line
```bash
python bepipred3_CLI.py -i ./example_antigens/antigens.fasta -o ./example_output/ -pred vt_pred -t 0.17
```
This will ESM-1b encode sequences antigens.fasta, make B-cell epitope predictions at a threshold of 0.17, and store it as a fasta file in example_output.

For more info, you can run
```bash
python bepipred3_CLI.py -h
```

### Creating your own setup 
You can also use bepipred3 in a more customized fashion and directly access ESM1-b encodings, model outputs etc. This is illustrated in DemoNoteBook.ipynb. 