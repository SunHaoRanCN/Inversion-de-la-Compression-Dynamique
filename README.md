# Inversion-de-la-Compression-Dynamique

Approche Hybride pour l'Inversion de la Compression Dynamique en Traitement Audio à l'aide de l'Apprentissage Profond
(Model and Deep learning based Dynamic Range Compression Inversion)

## Authors
- **Haoran Sun** <haoran.sun@etu-upsaclay.fr>
- **Dominique Fourer** <dominique.fourer@univ-evry.fr>
- **Hichem Maaref** <hichem.maaref@univ-evry.fr>

## Affiliations
Laboratoire IBISC (EA 4526), Univ. Evry Paris-Saclay, Évry-Courcouronnes, France

## Usage
Before training, please make sure your dataset is stored as:

```
├── dir/
│   ├── all
│   └── train
│   └── test
```

where the "all" folder should contain all the original chunks.

To generate an audio dataset, run

```
python3 data/Dataset.py
```

It will generate three folders: all, train, and test, in the folder you specified.

To train the AST model, run:

```
python3 run/classification.py
```  

To train the MEE model for DRC parameter estimation, run:

```
python3 run/regression.py
``` 

With Evaluation, the DRC inversion will be completed and the estimated original signals will be stored in the output_path.  

To compute the MSE, Mel and SISDR loss, run:

```
python3 run/compute_loss.py
```  

## Ref.
If you use this work, please cite:

Sun, Haoran, Dominique Fourer, and Hichem Maaref. "Model and Deep learning based Dynamic Range Compression Inversion." arXiv preprint arXiv:2411.04337 (2024).

@article{sun2024model,
  title={Model and Deep learning based Dynamic Range Compression Inversion},
  author={Sun, Haoran and Fourer, Dominique and Maaref, Hichem},
  journal={arXiv preprint arXiv:2411.04337},
  year={2024}
}

## License
Distributed under the MIT License.


## Acknowledgements
- [PyTorch](https://pytorch.org/)
