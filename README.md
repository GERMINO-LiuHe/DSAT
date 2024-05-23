# DSAT for 2D face alignment

This ia a PyTorch implemention for face alignment with Dynamic Semantic Aggregation Transformer (DSAT). We use the normalized mean errors (NME) to measure the landmark location performance. This work has achieved outstanding performance on 300W datasets.

## Install

* `Python 3`

* `Install PyTorch >= 0.4 following the official instructions (https://pytorch.org/).`

## data

You need to download images (e.g., 300W) from official websites and then put them into `data` folder for each dataset.

Your `data` directory should look like this:

````
DSAT
-- data
   |-- afw
   |-- helen
   |-- ibug
   |-- lfpw
````  

## Training and testing

* Train

```sh
python main.py 
```

* Run evaluation to get result.

```sh
python main.py --phase test
or
python demo.py
```

