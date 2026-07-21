# Raster Clustering Plugin for QGIS

This plugin executes clustering algorithms (K-Means and EM-GMM) on raster images using compiled binaries. It also provides a harmonization tool to align cluster labels across multiple results for visual comparison.

---

## Installation

1. **Compile the binaries** (CPU and GPU versions) and place them into the `plugin/bin/` directory.
2. Install the plugin in QGIS via the Plugin Manager (add the folder) or copy the whole `plugin/` directory into your QGIS profile’s `python/plugins/` folder.
3. Enable the plugin in QGIS. A new toolbar button and a menu entry under **Raster → Raster Clustering** will appear.

---

## Usage

### Opening the Plugin

Click the plugin icon or select **Raster → Raster Clustering**. The main dialog will open.

### 1. Select Execution Mode

- **CPU**: Uses the CPU-only binaries (`cpuKM.exe`, `cpuGMM.exe`).
- **CPU+GPU**: Uses the GPU-accelerated binaries (`gpuKM.exe`, `gpuGMM.exe`).

The binary path fields update automatically according to the selected mode. You can override the default paths if needed.

### 2. Choose Input Raster

Click **Browse…** and select the multiband GeoTIFF image you want to cluster.

### 3. Choose Algorithm

- **K-Means**: Standard K-Means clustering with random or K-Means++ initialization.
- **EM-GMM**: Expectation-Maximization Gaussian Mixture Model clustering.

Additional parameters appear/disappear depending on the algorithm.

### 4. Set Parameters

#### Common parameters

- **Number of clusters (k):** 2–12
- **Max iterations:** Maximum number of iterations (1–10000)
- **Convergence threshold:** Stop the algorithm when the change is below this value
- **Seed:** Random seed for reproducibility

#### K-Means specific

- **Initialisation:** Random or K-Means++

#### GMM specific

- **K-Means init iterations:** Number of K-Means iterations used to initialise the GMM.

### 5. Output Settings

- **Output folder:** Where the result files will be saved.
- **Output layer basename:** Prefix for the output file. The final name will be `<basename>_<algorithm>_<mode>.tif` (e.g. `clustering_kmeans_cpu.tif`).
- **Load result into QGIS automatically:** If checked, the resulting raster will be added to the opened project when finished.

### 6. Run

Click **Run**. The log window will show progress, and a progress bar will appear. You can cancel the process at any time. When completed, the output GeoTIFF (and a corresponding CSV of cluster statistics if created by the binary) will be saved in the output folder.

### 7. Result Harmonization (Optional)

If you have several clustering images (from different runs, algorithms, or parameters) and want to compare them with consistent colour coding, use the harmonization tool. All clustering results must have the same number of clusters in order for the harmonization to work.

1. Add images to the list using **Add…**.
2. Specify the **Original image** (the multiband raster used for clustering), by default it selects the image specified in the main tab.
3. Click **Harmonize**.

The tool computes centroids of each cluster, matches cluster labels across all images using the Hungarian algorithm, and saves new GeoTIFFs with the suffix `_harmonized.tif`. The first image in the list is used as the reference colour table.

---

## Adding a New Clustering Algorithm

The plugin can be extended with other clustering algorithms. Follow these steps:

### Binary Interface Requirements

Your binary must be callable from the command line with a fixed argument order and must produce:

- A GeoTIFF output named `<algorithm>_output.tif`).
- Optionally, a CSV file named `<algorithm>_output.csv`).

The output raster must contain integer class labels (pixel values `0 … k-1`) and should set the NoData value to `k` so that the plugin can retrieve the number of clusters from the image (for harmonization).

### Calling Convention for the Current Algorithms

Use these as a reference.

#### K-Means

```text
binary <input.tif> <k> <init> <max_iter> <convergence_threshold> <seed>
```

#### GMM

```text
binary <input.tif> <k> <max_iter> <convergence_threshold> <seed> <kmeans_init_iter>
```

You can design any argument list, but you must adapt the dialog code accordingly.

---

## Modifying the Plugin Code

All modifications are done in `dialog.py`. The easiest way is to add a new radio button for your algorithm and a new parameter panel.

### Step 1: Add Algorithm Radio Button

In `_build_ui()`, add a new `QRadioButton` inside the `algo_group` and add it to `self._algo_group`.

### Step 2: Add Parameters Widget

Create a new widget (like `gmm_init_row`) that contains the required controls for your algorithm. Initially hide it. Add it to the `param_layout`.

### Step 3: Update Visibility Logic

In `_update_param_visibility()`, show/hide your parameter widgets based on whether your radio button is checked.

### Step 4: Adjust Binary Selection and Command Construction

In `_on_run()`, extend the conditional chain that selects the binary:

```python
if self.rb_kmeans.isChecked():
    binary = self.gpu_km_bin_edit.text() if is_gpu else self.km_bin_edit.text()
elif self.rb_gmm.isChecked():
    binary = self.gpu_gmm_bin_edit.text() if is_gpu else self.gmm_bin_edit.text()
elif self.rb_myalgo.isChecked():
    binary = self.gpu_myalgo_bin_edit.text() if is_gpu else self.myalgo_bin_edit.text()
```

Then build the command list accordingly:

```python
cmd = [binary, img_path, ...]   # your arguments
```

### Step 5: Define Output File Handling

The plugin automatically renames the raw output (`kmeans_output.tif` or `gmm_output.tif`) to the formatted final name. If your binary outputs a file with a different name (e.g. `dbscan_output.tif`), modify the raw output path in `_on_done()`:

```python
if self.rb_myalgo.isChecked():
    raw_out = os.path.join(out_dir, "dbscan_output.tif")
```

Similarly, handle the CSV renaming there.

### Step 6: Add Binary Path Fields (Optional)

For CPU and GPU modes, you can either reuse the existing four binary fields (if your binary can replace an existing one) or add new ones in the `cpu_page` and `gpu_page` widgets. Add new `_path_row(...)` entries accordingly.

### Step 7: Register the New Binary Default Paths

In the constructor, you may add the default path for your binaries inside `_default_binary()`.

---

## Harmonization Compatibility

The harmonization tool automatically works with any labelled raster produced by the plugin, as long as the NoData value is set to the number of clusters (`k`). Make sure your binary does that.

---

## Troubleshooting

- **Binary not found:** Check the paths in the **Binary paths** group. By default they point to `plugin/bin/`.
- **Missing DLLs (Windows):** The plugin automatically adds `C:\OSGeo4W\apps\gdal-dev\bin` to the `PATH` for the subprocess. Ensure that the GDAL DLLs are present there.
- **Harmonization fails:** Verify that all images have the same number of clusters. The number of clusters is read from the NoData value of the first band.

For further assistance, consult the repository or contact the author.