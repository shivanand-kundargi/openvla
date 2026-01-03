# Bridge V2 Dataset Download

Quick script to download the Bridge V2 dataset for OpenVLA concept analysis.

## Usage

### Default location (~/datasets)

```bash
bash concept_analysis/download_bridge_v2.sh
```

### Custom location

```bash
bash concept_analysis/download_bridge_v2.sh /path/to/your/data/directory
```

## What it does

1. **Checks disk space** - Verifies you have ~134 GB available
2. **Downloads dataset** - Uses wget to download from Berkeley server (~124 GB)
3. **Renames correctly** - Automatically renames `bridge_dataset` → `bridge_orig`
4. **Verifies download** - Checks structure and reports statistics
5. **Shows next steps** - Provides commands to use the dataset

## Requirements

- ~134 GB free disk space
- Stable internet connection
- wget installed (usually pre-installed on Linux)

## Download time

Approximately **30-60 minutes** depending on your internet speed:
- 100 Mbps: ~3 hours
- 500 Mbps: ~40 minutes  
- 1 Gbps: ~20 minutes

## Resume interrupted download

If download is interrupted, just run the script again. It will resume from where it left off.

## After download

The dataset will be at:
- `~/datasets/bridge_orig` (default)
- Or your specified directory

Set the environment variable:
```bash
export DATA_ROOT=~/datasets  # or your custom path
```

Then run the concept analysis:
```bash
bash concept_analysis/run_rq1_language.sh
```
