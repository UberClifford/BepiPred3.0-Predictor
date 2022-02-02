#import pathlib
from pathlib import Path
import torch
import sys
from BP3.ESM1b.esm1b import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
ROOT_DIR = Path( Path(__file__).parent.resolve() )


def ESM1b_encode_fasta_file(esm1b_encoding_save_dir):

    fasta_file = esm1b_encoding_save_dir / "antigens.fasta" 

    toks_per_batch = 4096
    include = ["per_tok"]
    repr_layers = [-1]
    nogpu = False
    truncate = True
    model_location = ROOT_DIR / "models" / "esm1b_t33_650M_UR50S.pt"
  
    model, alphabet = pretrained.load_model_and_alphabet( model_location )
    model.eval()
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read {fasta_file} with {len(dataset)} sequences")
    esm1b_encoding_save_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in include
    
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
 
            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if truncate:
                print("Truncating sequences longer than 1024")
                toks = toks[:, :1022]
 
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
 
            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")
 
            for i, label in enumerate(labels):
                output_file = esm1b_encoding_save_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                result["representations"] = {layer: t[i, 1 : len(strs[i]) + 1].clone() for layer, t in representations.items()}
 
                torch.save(
                    result,
                    output_file,
                )