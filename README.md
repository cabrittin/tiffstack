# tiffstack

Analysis of TiFF Stack files

See 'example_data_dir' for how to set up directories for your images

## Setting up python environment

```
pip install -r requirements.txt
```
For each dataset, preprocess requires 3 steps. 

## Step 1: Set the ROIs

```
python scripts/run_sequence.py set_roi example_data_dir
```

Left click on the each embryo. Press 'q' when done. 

## Step 2: Check the ROIs

Two ways:
```
python scripts/run_sequence.py viz_roi example_data_dir
```
or
```
python scripts/run_sequence.py viz_roi_single example_data_dir --roi_index=0,1
```
where 'roi_index' specifies which rois to view. 

## Step 3: Extract the ROIs. 

Two ways:
```
python scripts/run_sequence.py extract_roi example_data_dir
```
or with parallel processing
```
python scripts/run_sequence.py extract_roi_mp example_data_dir
```

