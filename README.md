# WEmarker

WEmarker: breast cancer-specific prognostic analysis with weighted multiplex network embedding

## Network Enhancement

`./Network Enhancement/main.m` is the main script to run NE. The format of the input is a collar matrix. 

## Weighted Multiplex Network Embedding(WMNE)

### Input

File input format of the WMNE algorithm：

```
Layer  head  tail  weight
  l1    n1    n2     w1
  l2    n3    n4     w2
  l3    n5    n6     w3
            .
            .
            .
```

The specific form of input files just like: `./data/Fusion6Net.txt`

### Run embedding

```python
python train_model.py
```

- The embedding vector is saved `./model`. 
- A demo version of the embedding vector is already available under this file.

### Acknowledgment

WMNE is developed based on [MNE](https://github.com/HKUST-KnowComp/MNE), for which we are sincerely grateful.

## Get biomarker

```python
python GetBiomarker.py
```

- Make sure you get the embedding vector before then.
-  `./allscore/all_score_GSE1456.json` holds the initial score for each gene in *GSE1456*, you can replace it with your own data.
- The gene ranking file obtained after running is saved in `./Biomarker`.

