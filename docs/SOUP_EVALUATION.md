# Soup: future Qwen-teacher option

Repository: [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup) (Apache-2.0).

## Decision

Do not use Soup for the current Ettin 150M Anarkali release. Anarkali is a packed encoder choice model, while Soup fine-tunes causal LLMs. It cannot train or serve the recovered `best.pt` checkpoint.

Consider Soup in the Qwen teacher phase after the 150M release benchmark is complete. Its possible role is to QLoRA fine-tune a Qwen model on the local 4 GB GTX 1650, then use that model to generate or relabel decision data before distilling into Anarkali.

## Constraints

- Soup documents Python 3.10 through 3.12 support. The current project environment is Python 3.14, so create a separate Python 3.12 environment if this work starts.
- Keep Qwen training data, teacher outputs, and Anarkali held-out test splits separate. Never tune the teacher on locked test data.
- Treat Soup's 4 GB layer-streaming claim as a candidate workflow that needs a local reproduction before relying on it.

## Next step

After publishing the measured 150M model card, run Soup's hardware preflight and a small Qwen QLoRA pilot. Compare the resulting teacher with the existing 400M teacher on the locked calibration and development splits before using it for a distillation run.
