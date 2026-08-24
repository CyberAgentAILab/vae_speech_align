# JVS Alignment Example

Train a VAE-based speech alignment model on the [JVS (Japanese Versatile Speech) corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus).

## Setup

1. Download the JVS corpus (`jvs_ver1`) from the official site.
2. Prepare the data:
   ```bash
   uv run bash prepare_data.sh /path/to/jvs_ver1
   ```
3. Train:
   ```bash
   uv run bash run_train.sh
   ```
4. Run alignment:
   ```bash
   uv run bash run_align.sh
   ```

## License

`data/phoneme.tar.gz` contains phoneme labels derived from the JVS corpus.
The text data of the JVS corpus originates from the [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
This file is distributed under the same CC BY-SA 4.0 license.

## References

- Takamichi, S., Mitsui, K., Saito, Y., Koriyama, T., Tanji, N., & Saruwatari, H. (2020). JVS corpus: free Japanese multi-speaker voice corpus. *arXiv preprint arXiv:1908.06248*.
