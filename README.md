# Clustering Acceleration

Accelerated clustering algorithms (K-Means and Gaussian Mixture Models) for raster images, with a QGIS plugin for easy execution, visualization, and result harmonization.

---

## Project Overview

This repository provides:

- **CPU implementations** of K-Means and EM-GMM in C.
- **GPU-accelerated implementations** of the same algorithms using CUDA.
- A **QGIS plugin** that runs the compiled binaries on multiband GeoTIFF rasters and displays the output directly in QGIS and can align cluster labels across different results using the Hungarian algorithm, enabling easier visual comparison.

All binaries accept command-line arguments for full parameter control and produce labelled GeoTIFFs with colour tables, ready for analysis, as well as a raw CSV with the labelled pixels.

---

## Directory Structure

```text
clustering-acceleration/
├── clustering/                     # Algorithm source code
│   ├── common.c / common.h         # Shared utilities (file I/O, error handling, etc.)
│   ├── cpu_KMeans.c                # CPU K-Means implementation
│   ├── cpu_GMM.c                   # CPU EM-GMM implementation
│   ├── gpu_KMeans.cu               # GPU (CUDA) K-Means
│   └── gpu_GMM.cu                  # GPU (CUDA) EM-GMM
├── plugin/                         # QGIS plugin
│   ├── __init__.py                 # Plugin initialisation
│   ├── dialog.py                   # Main dialog and logic
│   ├── metadata.txt                # Plugin metadata
│   ├── plugin.py                   # Plugin main class
│   ├── bin/                        # Folder for compiled binaries
│   │   └── bin.md                  # Instructions for placing the executables
│   └── plugin.md                   # Plugin user guide
├── LICENSE
└── README.md
```

---

## Dependencies

### Building the Binaries

- **C compiler** (GCC, Clang, MSVC, etc.) for CPU code.
- **CUDA Toolkit** (`nvcc`) for GPU code.
- **GDAL library** (`libgdal`) – the binaries read and write GeoTIFF files using GDAL's C API.

### Running the Plugin

- **QGIS ≥ 4.0** (could work on some versions of QGIS 3).
- **Python packages**:
  - `numpy`
  - `scipy` (for the harmonization tool)
  - `gdal` Python bindings (usually bundled with QGIS)
- The compiled binaries are to be placed in `plugin/bin/` (see installation).

---

## Building the Binaries

### 1. Install GDAL Development Packages

- **Windows (OSGeo4W):** `gdal-dev` package provides headers and libraries.
- **Linux:** `libgdal-dev` (Debian/Ubuntu) or `gdal-devel` (Fedora).

### 2. Install the CUDA Toolkit

Required only for building the GPU implementations.

### 3. Compile the Source Files

#### CPU K-Means

```bash
gcc -O3 clustering/cpu_KMeans.c clustering/common.c -o plugin/bin/cpuKM.exe -lgdal
```

#### CPU GMM

```bash
gcc -O3 clustering/cpu_GMM.c clustering/common.c -o plugin/bin/cpuGMM.exe -lgdal
```

#### GPU K-Means

```bash
nvcc -O3 clustering/gpu_KMeans.cu clustering/common.c -o plugin/bin/gpuKM.exe -lgdal
```

#### GPU GMM

```bash
nvcc -O3 clustering/gpu_GMM.cu clustering/common.c -o plugin/bin/gpuGMM.exe -lgdal
```

### Windows (OSGeo4W)

You may need to specify the GDAL include and library paths explicitly:

```bash
gcc ... \
    -I"C:/OSGeo4W/apps/gdal-dev/include" \
    -L"C:/OSGeo4W/apps/gdal-dev/lib" \
    -lgdal
```

The output executables must be placed (or built directly) into `plugin/bin/`.

---

## Installing the QGIS Plugin

1. Compile the four binaries as described above and ensure they are inside `plugin/bin/`:

   - `cpuKM.exe`
   - `gpuKM.exe`
   - `cpuGMM.exe`
   - `gpuGMM.exe`

2. Copy (or symlink) the `plugin/` folder into your QGIS profile's plugins directory `QGIS\QGIS3\profiles\default\python\plugins\`.

3. Enable the plugin in QGIS:

   - Open **Plugins → Manage and Install Plugins…**
   - Find **Raster Clustering**
   - Check the box to enable it.

A new toolbar button and a menu entry under **Raster** will appear.

---

## Usage

For a complete description, refer to **`plugin/plugin.md`**.

In summary:

1. Launch the plugin from the toolbar or **Raster → Raster Clustering**.
2. Select the execution mode (**CPU** or **CPU+GPU**). The corresponding binary paths are filled automatically but can be overridden.
3. Choose the input multiband GeoTIFF.
4. Select the clustering algorithm (**K-Means** or **EM-GMM**) and configure its parameters:
   - Number of clusters
   - Maximum iterations
   - Convergence threshold
   - Random seed
   - Algorithm-specific options
5. Choose the output folder and base name. The output raster will be saved as:

   ```text
   <basename>_<algorithm>_<mode>.tif
   ```

6. Click **Run**. Progress is displayed in the log window and the process can be cancelled at any time.
7. Optionally, use the **Harmonization** tab to align cluster labels across multiple clustering outputs for direct comparison. The tool requires the original input raster and uses the Hungarian algorithm to match cluster centroids.

---

## Adding New Algorithms

The plugin is designed to be extensible.

To integrate a new clustering algorithm (e.g. DBSCAN or hierarchical clustering), see the detailed guide in **`plugin/plugin.md`** under **"Adding a New Clustering Algorithm"**.

In summary:

- Write a compliant binary that reads a GeoTIFF.
- Produce a labelled output raster.
- Set the raster **NoData** value equal to the number of clusters.
- Add a radio button for the new algorithm in the plugin dialog.
- Add the algorithm-specific parameters.
- Construct the appropriate command-line arguments.
- Handle the output file renaming.

The harmonization tool will automatically work with any output that follows the NoData convention.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

**Author:** Adam El Fakhouri

For issues or suggestions, please open an issue on GitHub.
