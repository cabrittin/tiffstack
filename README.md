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
python scripts/run_sequence.py extract_roi_mp example_data_dir --num_jobs=6
```
where 'num_jobs' is the number of CPUs. 

## Step 4: Create masks. 

```
python scripts/run_analysis.py create_mask example_data_dir
```

## Step 5: Check that masks segment the embryos

```
python scripts/run_analysis.py viz_mask example_data_dir --roi_index=0,1,2
```

## Step 6: Process proportion of pixel changes

```
python scripts/run_analysis.py compute_pixel_change data/ac_12hr_6/
```
Output files saved to performance directory

## Step 7: Pixel value in the embryo relative to background (optional)
```
python scripts/run_analysis.py compute_pixel_distribution data/ac_12hr_6/
```
Output files saved to performance directory


