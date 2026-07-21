import os
import sys
import subprocess
import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
    QLineEdit, QFileDialog, QTextEdit, QProgressBar,
    QRadioButton, QButtonGroup, QMessageBox,
    QCheckBox, QStackedWidget, QWidget, QListWidget,
    QAbstractItemView, QScrollArea
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QFont
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog, Qgis
from scipy.optimize import linear_sum_assignment
from osgeo import gdal


class WorkerThread(QThread):
    """
    Hilo que ejecuta el binario de clustering en segundo plano
    para no congelar la interfaz de QGIS.
    """
    output_line = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, cmd, cwd, output_tif):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.output_tif = output_tif
        self.proc = None
        self._cancelled = False

    def run(self):
        try:
            env = os.environ.copy()
            if sys.platform == "win32":
                osgeo_bin = r"C:\OSGeo4W\bin"
                gdal_bin = r"C:\OSGeo4W\apps\gdal-dev\bin"
                env["PATH"] = gdal_bin + os.pathsep + osgeo_bin + os.pathsep + env.get("PATH", "")

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

            self.proc = subprocess.Popen(
                self.cmd, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, creationflags=creationflags
            )
            for line in self.proc.stdout:
                self.output_line.emit(line.rstrip())
            self.proc.wait()

            if self._cancelled:
                self.finished_err.emit("Cancelled by user.")
            elif self.proc.returncode != 0:
                self.finished_err.emit(f"Process exited with code {self.proc.returncode}.")
            else:
                self.finished_ok.emit(self.output_tif)
        except Exception as e:
            self.finished_err.emit(str(e))

    def cancel(self):
        self._cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    self.proc.terminate()
            except Exception:
                pass

class ClusteringDialog(QDialog):
    """
    Dialog principal del plugin. Permite seleccionar algoritmo (K-Means o GMM),
    modo de ejecucion (CPU/GPU), parametros y armonizar resultados.
    """
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.worker = None
        self.setWindowTitle("Raster Clustering")
        self.setMinimumWidth(560)
        self._build_ui()
        self._cancel_requested = False

    def _build_ui(self):
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setSpacing(10)

        exec_group = QGroupBox("Execution mode")
        exec_layout = QHBoxLayout(exec_group)
        self.exec_cpu_rb = QRadioButton("CPU")
        self.exec_gpu_rb = QRadioButton("CPU+GPU")
        self.exec_cpu_rb.setChecked(True)
        self.exec_mode_group = QButtonGroup(self)
        self.exec_mode_group.addButton(self.exec_cpu_rb, 0)
        self.exec_mode_group.addButton(self.exec_gpu_rb, 1)
        exec_layout.addWidget(self.exec_cpu_rb)
        exec_layout.addWidget(self.exec_gpu_rb)
        exec_layout.addStretch()
        root.addWidget(exec_group)

        bin_group = QGroupBox("Binary paths")
        bin_group_layout = QVBoxLayout(bin_group)

        self.bin_stack = QStackedWidget()

        cpu_page = QWidget()
        cpu_layout = QVBoxLayout(cpu_page)
        cpu_layout.setContentsMargins(0, 0, 0, 0)

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        plugin_dir = os.path.join(plugin_dir, "bin")

        self.km_bin_edit = self._path_row(
            cpu_layout,
            "CPU K-Means binary:",
            self._default_binary(plugin_dir, "cpuKM.exe")
        )
        self.gmm_bin_edit = self._path_row(
            cpu_layout,
            "CPU EM-GMM binary:",
            self._default_binary(plugin_dir, "cpuGMM.exe")
        )
        self.bin_stack.addWidget(cpu_page)

        gpu_page = QWidget()
        gpu_layout = QVBoxLayout(gpu_page)
        gpu_layout.setContentsMargins(0, 0, 0, 0)

        self.gpu_km_bin_edit = self._path_row(
            gpu_layout,
            "GPU K-Means binary:",
            self._default_binary(plugin_dir, "gpuKM.exe")
        )
        self.gpu_gmm_bin_edit = self._path_row(
            gpu_layout,
            "GPU EM-GMM binary:",
            self._default_binary(plugin_dir, "gpuGMM.exe")
        )
        self.bin_stack.addWidget(gpu_page)

        bin_group_layout.addWidget(self.bin_stack)
        root.addWidget(bin_group)

        img_group = QGroupBox("Input raster")
        img_layout = QVBoxLayout(img_group)
        self.img_edit = self._path_row(img_layout, "Image (.tif):", None, is_file=True)
        root.addWidget(img_group)

        algo_group = QGroupBox("Algorithm")
        algo_layout = QHBoxLayout(algo_group)
        self.rb_kmeans = QRadioButton("K-Means")
        self.rb_gmm = QRadioButton("EM-GMM")
        self.rb_kmeans.setChecked(True)
        self._algo_group = QButtonGroup(self)
        self._algo_group.addButton(self.rb_kmeans)
        self._algo_group.addButton(self.rb_gmm)
        algo_layout.addWidget(self.rb_kmeans)
        algo_layout.addWidget(self.rb_gmm)
        root.addWidget(algo_group)

        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)

        k_row = QHBoxLayout()
        k_row.addWidget(QLabel("Number of clusters (k):"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 12)
        self.k_spin.setValue(4)
        self.k_spin.setFixedWidth(70)
        k_row.addWidget(self.k_spin)
        k_row.addStretch()
        param_layout.addLayout(k_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 10000)
        self.iter_spin.setValue(100)
        self.iter_spin.setFixedWidth(100)
        iter_row.addWidget(self.iter_spin)
        iter_row.addStretch()
        param_layout.addLayout(iter_row)

        conv_row = QHBoxLayout()
        conv_row.addWidget(QLabel("Convergence threshold:"))
        self.conv_spin = QDoubleSpinBox()
        self.conv_spin.setDecimals(6)
        self.conv_spin.setRange(0.000001, 1.0)
        self.conv_spin.setSingleStep(0.0001)
        self.conv_spin.setValue(0.001)
        self.conv_spin.setFixedWidth(120)
        conv_row.addWidget(self.conv_spin)
        conv_row.addStretch()
        param_layout.addLayout(conv_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999999)
        self.seed_spin.setValue(123456789)
        self.seed_spin.setFixedWidth(120)
        seed_row.addWidget(self.seed_spin)
        seed_row.addStretch()
        param_layout.addLayout(seed_row)

        self.init_row_widget = QHBoxLayout()
        self.init_row_widget.addWidget(QLabel("Initialisation:"))
        self.init_combo = QComboBox()
        self.init_combo.addItem("Random", "random")
        self.init_combo.addItem("K-Means++", "pp")
        self.init_combo.setFixedWidth(130)
        self.init_row_widget.addWidget(self.init_combo)
        self.init_row_widget.addStretch()
        param_layout.addLayout(self.init_row_widget)

        self.gmm_init_row = QHBoxLayout()
        self.gmm_init_row.addWidget(QLabel("K-Means init iterations:"))
        self.kmeans_init_spin = QSpinBox()
        self.kmeans_init_spin.setRange(1, 1000)
        self.kmeans_init_spin.setValue(10)
        self.kmeans_init_spin.setFixedWidth(100)
        self.gmm_init_row.addWidget(self.kmeans_init_spin)
        self.gmm_init_row.addStretch()
        param_layout.addLayout(self.gmm_init_row)

        root.addWidget(param_group)

        harm_group = QGroupBox("Result harmonization")
        harm_layout = QVBoxLayout(harm_group)

        list_row = QHBoxLayout()
        self.harm_list = QListWidget()
        self.harm_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.harm_add_btn = QPushButton("Add...")
        self.harm_remove_btn = QPushButton("Erase selection")
        list_btn_layout = QVBoxLayout()
        list_btn_layout.addWidget(self.harm_add_btn)
        list_btn_layout.addWidget(self.harm_remove_btn)
        list_btn_layout.addStretch()
        list_row.addWidget(self.harm_list)
        list_row.addLayout(list_btn_layout)
        harm_layout.addLayout(list_row)

        orig_row = QHBoxLayout()
        orig_row.addWidget(QLabel("Original image:"))
        self.harm_orig_edit = QLineEdit()
        self.harm_orig_browse = QPushButton("Browse...")
        orig_row.addWidget(self.harm_orig_edit)
        orig_row.addWidget(self.harm_orig_browse)
        harm_layout.addLayout(orig_row)

        self.harm_button = QPushButton("Harmonize")
        harm_layout.addWidget(self.harm_button)

        root.addWidget(harm_group)

        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        self.out_edit = self._path_row(
            out_layout, "Output folder:",
            os.path.expanduser("~"), is_dir=True
        )
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Output layer base name:"))
        self.name_edit = QLineEdit("clustering")
        name_row.addWidget(self.name_edit)
        out_layout.addLayout(name_row)

        self.load_cb = QCheckBox("Load result into QGIS automatically")
        self.load_cb.setChecked(True)
        out_layout.addWidget(self.load_cb)
        root.addWidget(out_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Monospace", 8))
        self.log_box.setMinimumHeight(160)
        log_layout.addWidget(self.log_box)
        root.addWidget(log_group)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.close_btn = QPushButton("Close")
        self.cancel_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self.rb_kmeans.toggled.connect(self._update_param_visibility)
        self.exec_mode_group.buttonToggled.connect(self._update_param_visibility)
        self.run_btn.clicked.connect(self._on_run)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.close_btn.clicked.connect(self.close)
        self.harm_add_btn.clicked.connect(self._add_harm_files)
        self.harm_remove_btn.clicked.connect(self._remove_harm_files)
        self.harm_orig_browse.clicked.connect(lambda: self._browse_file(self.harm_orig_edit))
        self.harm_button.clicked.connect(self._on_harmonize)

        self._update_param_visibility()

        scroll.setWidget(container)
        dialog_layout.addWidget(scroll)

        self.resize(650, 750)
        self.setMinimumWidth(580)

    def _path_row(self, parent_layout, label, default="", is_file=False, is_dir=False):
        """Crea una fila con etiqueta, campo de texto y boton Browse."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(160)
        edit = QLineEdit(default or "")
        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        row.addWidget(lbl)
        row.addWidget(edit)
        row.addWidget(btn)
        parent_layout.addLayout(row)

        if is_file:
            btn.clicked.connect(lambda: self._browse_file(edit))
        elif is_dir:
            btn.clicked.connect(lambda: self._browse_dir(edit))
        else:
            btn.clicked.connect(lambda: self._browse_file(edit, exe=True))

        return edit

    def _get_output_path(self):
        """
        Construye la ruta del archivo de salida con el nombre base,
        el sufijo del algoritmo y el modo de ejecucion.
        """
        base = self.name_edit.text().strip() or "clustering"
        out_dir = self.out_edit.text().strip()
        is_kmeans = self.rb_kmeans.isChecked()
        is_gpu = self.exec_mode_group.checkedId() == 1
        algo = "kmeans" if is_kmeans else "gmm"
        mode = "gpu" if is_gpu else "cpu"
        filename = f"{base}_{algo}_{mode}.tif"
        return os.path.normpath(os.path.join(out_dir, filename))

    def _default_binary(self, base_dir, name):
        full = os.path.normpath(os.path.join(base_dir, name))
        if os.path.isfile(full):
            return full.replace("\\", "/")
        return name

    def _browse_file(self, edit, exe=False):
        if exe:
            path, _ = QFileDialog.getOpenFileName(self, "Select binary", "", "All files (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select raster", "", "GeoTIFF (*.tif *.tiff);;All files (*)"
            )
        if path:
            edit.setText(path)

    def _browse_dir(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Select output folder", edit.text())
        if path:
            edit.setText(path)

    def _update_param_visibility(self):
        """
        Muestra u oculta los parametros especificos segun el algoritmo
        (K-Means o GMM) y el modo (CPU/GPU) seleccionados.
        """
        is_kmeans = self.rb_kmeans.isChecked()
        is_gpu = self.exec_mode_group.checkedId() == 1

        self.bin_stack.setCurrentIndex(1 if is_gpu else 0)

        for i in range(self.init_row_widget.count()):
            item = self.init_row_widget.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(is_kmeans)

        for i in range(self.gmm_init_row.count()):
            item = self.gmm_init_row.itemAt(i)
            if item and item.widget():
                item.widget().setVisible(not is_kmeans)

    def _on_run(self):
        """
        Valida los parametros, construye la linea de comandos y lanza
        el hilo de ejecucion.
        """
        self._cancel_requested = False

        img_path = self.img_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        k = self.k_spin.value()
        max_iter = self.iter_spin.value()
        convergence = self.conv_spin.value()
        seed = self.seed_spin.value()

        is_kmeans = self.rb_kmeans.isChecked()
        is_gpu = self.exec_mode_group.checkedId() == 1

        if is_kmeans:
            if is_gpu:
                binary = self.gpu_km_bin_edit.text().strip()
            else:
                binary = self.km_bin_edit.text().strip()
            algo = "kmeans"
        else:
            if is_gpu:
                binary = self.gpu_gmm_bin_edit.text().strip()
            else:
                binary = self.gmm_bin_edit.text().strip()
            algo = "gmm"

        self._log(f"Checking binary: '{binary}' (exists: {os.path.isfile(binary)})")
        errors = []
        if not binary or not os.path.isfile(binary):
            errors.append("Binary not found — check the binary path.")
        if not img_path or not os.path.isfile(img_path):
            errors.append("Input raster not found.")
        if not out_dir or not os.path.isdir(out_dir):
            errors.append("Output folder does not exist.")

        if errors:
            QMessageBox.critical(self, "Input error", "\n".join(errors))
            return

        output_tif = self._get_output_path()

        if algo == "kmeans":
            init_str = self.init_combo.currentData()
            cmd = [binary, img_path, str(k), init_str,
                    str(max_iter), str(convergence), str(seed)]
        else:
            kmeans_init_iter = self.kmeans_init_spin.value()
            cmd = [binary, img_path, str(k),
                    str(max_iter), str(convergence), str(seed),
                    str(kmeans_init_iter)]

        self._log(f"Running: {' '.join(cmd)}")
        self._set_running(True)

        self.worker = WorkerThread(cmd, out_dir, output_tif)
        self.worker.output_line.connect(self._log)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.finished_err.connect(self._on_error)
        self.worker.start()

    def _on_done(self, output_tif):
        """
        Se ejecuta al terminar correctamente el binario.
        Renombra el archivo de salida al nombre deseado y opcionalmente lo carga en QGIS.
        """
        self._set_running(False)
        out_dir = os.path.normpath(self.out_edit.text().strip())

        if self.rb_kmeans.isChecked():
            raw_out = os.path.join(out_dir, "kmeans_output.tif")
        else:
            raw_out = os.path.join(out_dir, "gmm_output.tif")

        if not os.path.isfile(raw_out):
            self._log(f"WARNING: expected output file not found: {raw_out}")
            if os.path.isfile(output_tif):
                raw_out = output_tif
            else:
                QMessageBox.warning(self, "Missing output",
                                    f"The algorithm did not create the expected output file:\n{raw_out}")
                return

        # Si ya existe el archivo final, se elimina junto con sus capas en QGIS
        if os.path.isfile(output_tif):
            self._remove_layer_by_path(output_tif)
            try:
                os.remove(output_tif)
            except OSError as e:
                self._log(f"Warning: could not delete existing file: {e}")

        try:
            if raw_out != output_tif:
                os.rename(raw_out, output_tif)
                self._log(f"Output saved to: {output_tif}")
            else:
                self._log(f"Output already at: {output_tif}")
        except PermissionError as e:
            self._log(f"ERROR: Cannot rename output file.\n{e}")
            QMessageBox.critical(self, "Permission Error",
                                 f"Cannot rename the output file.\n\n{e}")
            return

        if self.load_cb.isChecked():
            layer_name = os.path.splitext(os.path.basename(output_tif))[0]
            self._load_layer(output_tif, layer_name)

        raw_csv = os.path.join(out_dir, "kmeans_output.csv" if self.rb_kmeans.isChecked() else "gmm_output.csv")
        if os.path.isfile(raw_csv):
            csv_out = os.path.splitext(output_tif)[0] + ".csv"
            try:
                os.replace(raw_csv, csv_out)
                self._log(f"Output saved to: {csv_out}")
            except OSError as e:
                self._log(f"Warning: could not rename CSV output: {e}")

    def _on_error(self, msg):
        self._set_running(False)
        self._log(f"ERROR: {msg}")
        if not self._cancel_requested:
            QMessageBox.critical(self, "Execution error", msg)

    def _on_cancel(self):
        if self.worker and self.worker.isRunning():
            self._cancel_requested = True
            self.worker.cancel()
            self.worker.wait(5000)
            self._set_running(False)

    def _load_layer(self, tif_path, layer_name):
        """Carga la capa raster resultante en el proyecto de QGIS."""
        if not os.path.isfile(tif_path):
            self._log(f"Cannot load layer — file not found: {tif_path}")
            return

        layer = QgsRasterLayer(tif_path, layer_name)
        if not layer.isValid():
            self._log("Layer loaded but reported invalid by QGIS.")
            QMessageBox.warning(self, "Warning", "Layer loaded but may be invalid.")
        else:
            QgsProject.instance().addMapLayer(layer)
            self.iface.setActiveLayer(layer)
            self.iface.zoomToActiveLayer()
            self._log(f"Layer '{layer_name}' added to QGIS.")

    def _remove_layer_by_path(self, path):
        """Elimina del proyecto QGIS todas las capas del fichero que se tiene que sobreescribir."""
        norm = os.path.normpath(path)
        layers_to_remove = []
        for layer in QgsProject.instance().mapLayers().values():
            if os.path.normpath(layer.source()) == norm:
                layers_to_remove.append(layer.id())
        if layers_to_remove:
            QgsProject.instance().removeMapLayers(layers_to_remove)
            self._log(f"Removed {len(layers_to_remove)} existing layer(s) to overwrite output.")
            return True
        return False

    def _log(self, text):
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())
        QgsMessageLog.logMessage(text, "Clustering", Qgis.Info)

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.progress.setVisible(running)

    # Armonizacion de resultados
    def _add_harm_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select clustered images",
                                                "", "GeoTIFF (*.tif *.tiff)")
        for p in paths:
            if not any(self.harm_list.item(i).text() == p for i in range(self.harm_list.count())):
                self.harm_list.addItem(p)

    def _remove_harm_files(self):
        for item in self.harm_list.selectedItems():
            self.harm_list.takeItem(self.harm_list.row(item))

    def _get_image_k(self, path):
        """Obtiene el numero de clusters (k) almacenado como valor NoData en el raster."""
        ds = gdal.Open(path)
        if not ds:
            return None
        ndv = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        if ndv is None:
            return None
        return int(ndv)

    def _on_harmonize(self):
        """
        Inicia el proceso de armonizacion: hace corresponder las etiquetas
        de varias segmentaciones a una referencia común.
        """
        original = self.harm_orig_edit.text().strip() or self.img_edit.text().strip()
        if not original or not os.path.isfile(original):
            QMessageBox.critical(self, "Error", "Specify a valid original image.")
            return

        files = [self.harm_list.item(i).text() for i in range(self.harm_list.count())]
        if len(files) < 2:
            QMessageBox.critical(self, "Error", "Select at least two images to harmonize.")
            return

        k = None
        for f in files:
            k_img = self._get_image_k(f)
            if k_img is None:
                QMessageBox.critical(self, "Error",
                    f"Could not determine number of clusters in {os.path.basename(f)}.")
                return
            if k is None:
                k = k_img
            elif k_img != k:
                QMessageBox.critical(self, "Error",
                    f"Images have different numbers of clusters ({k} vs {k_img} in {os.path.basename(f)}).\n"
                    "All images must have the same number of clusters.")
                return

        self._log(f"Detected k = {k} from the segmentation images.")

        self.harm_button.setEnabled(False)
        self.progress.setVisible(True)

        self.worker = HarmonizationWorker(original, files, k, out_suffix="_harmonized")
        self.worker.progress_text.connect(self._log)
        self.worker.finished_ok.connect(self._on_harm_done)
        self.worker.finished_err.connect(self._on_harm_error)
        self.worker.start()

    def _on_harm_done(self, output_paths):
        self.progress.setVisible(False)
        self.harm_button.setEnabled(True)
        self._log("Harmonization success. Generated files:")
        for p in output_paths:
            self._log(f"  {p}")
        if self.load_cb.isChecked():
            for p in output_paths:
                name = os.path.splitext(os.path.basename(p))[0]
                self._load_layer(p, name)

    def _on_harm_error(self, msg):
        self.progress.setVisible(False)
        self.harm_button.setEnabled(True)
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error while harmonizing", msg)


class HarmonizationWorker(QThread):
    """
    Hilo que realiza la armonizacion de etiquetas entre varias imagenes de
    clustering, usando el algoritmo Hungaro para emparejar centroides.
    """
    progress_text = pyqtSignal(str)
    finished_ok = pyqtSignal(list)
    finished_err = pyqtSignal(str)

    def __init__(self, original_path, file_paths, k, out_suffix="_harmonized"):
        super().__init__()
        self.original_path = original_path
        self.file_paths = file_paths
        self.k = k
        self.out_suffix = out_suffix

    def run(self):
        try:
            ds = gdal.Open(self.original_path)
            if not ds:
                raise IOError("Couldn't open original image")
            n_bands = ds.RasterCount
            rows, cols = ds.RasterYSize, ds.RasterXSize
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            bands = np.zeros((n_bands, rows, cols), dtype=np.float32)
            for b in range(n_bands):
                bands[b] = ds.GetRasterBand(b+1).ReadAsArray()
            ds = None

            seg_data = []
            for f in self.file_paths:
                ds_seg = gdal.Open(f)
                if not ds_seg:
                    raise IOError(f"Couldn't open {f}")
                labels = ds_seg.GetRasterBand(1).ReadAsArray()
                ndv = ds_seg.GetRasterBand(1).GetNoDataValue()
                if ndv is None:
                    ndv = self.k
                ds_seg = None
                seg_data.append({'path': f, 'labels': labels, 'nodata': ndv})

            centroids_list = []
            for seg in seg_data:
                centroids = self._compute_centroids(bands, seg['labels'], seg['nodata'])
                centroids_list.append(centroids)

            ref_idx = np.random.randint(0, len(seg_data))
            ref_centroids = centroids_list[ref_idx]
            ref_seg = seg_data[ref_idx]
            ref_ds = gdal.Open(ref_seg['path'])
            ct = ref_ds.GetRasterBand(1).GetColorTable()
            ref_ct = ct.Clone() if ct is not None else None
            ref_ds = None
            self.progress_text.emit(f"Reference image: {os.path.basename(ref_seg['path'])}")

            output_paths = []
            for i, seg in enumerate(seg_data):
                if i == ref_idx:
                    out_path = self._get_output_path(seg['path'])
                    self._save_image(out_path, seg['labels'], gt, proj, seg['nodata'], color_table=ref_ct)
                    output_paths.append(out_path)
                    continue
                mapping = self._match_centroids(ref_centroids, centroids_list[i])
                new_labels = self._apply_mapping(seg['labels'], mapping, seg['nodata'])
                out_path = self._get_output_path(seg['path'])
                self._save_image(out_path, new_labels, gt, proj, seg['nodata'], color_table=ref_ct)
                output_paths.append(out_path)
                self.progress_text.emit(f"Harmonized {os.path.basename(seg['path'])} -> {os.path.basename(out_path)}")

            self.finished_ok.emit(output_paths)

        except Exception as e:
            self.finished_err.emit(str(e))

    def _compute_centroids(self, bands, labels, nodata):
        """
        Calcula los centroides de cada cluster usando las bandas originales,
        ignorando los pixeles con valor nodata.
        """
        centroids = np.zeros((self.k, bands.shape[0]), dtype=np.float64)
        counts = np.zeros(self.k, dtype=np.int32)
        mask = (labels != nodata)
        for c in range(self.k):
            cmask = mask & (labels == c)
            if np.any(cmask):
                for b in range(bands.shape[0]):
                    centroids[c, b] = np.mean(bands[b][cmask])
            counts[c] = np.sum(cmask)
        return centroids

    def _match_centroids(self, ref_centroids, other_centroids):
        """
        Empareja los centroides de dos segmentaciones usando el algoritmo Hungaro.u
        """
        k = self.k
        dist = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                dist[i, j] = np.sqrt(np.sum((ref_centroids[i] - other_centroids[j])**2))
        row_ind, col_ind = linear_sum_assignment(dist)u
        mapping = np.full(k, -1, dtype=int)
        for r, c in zip(row_ind, col_ind):
            mapping[c] = ru
        return mapping

    def _apply_mapping(self, labels, mapping, nodata):
        """
        Aplica el mapeo de etiquetas old->new, manteniendo el valor nodata.
        """
        new_labels = np.full(labels.shape, nodata, dtype=np.uint8)
        valid_mask = (labels != nodata)
        for old, new in enumerate(mapping):
            if new == -1:
                continue
            new_labels[(labels == old) & valid_mask] = new
        return new_labels

    def _save_image(self, out_path, labels, gt, proj, nodata, color_table=None):
        """Guarda el raster de etiquetas en formato GeoTIFF con tabla de color."""
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError as e:
                raise IOError(f"Cannot overwrite {out_path}: {e}. It may be open in QGIS.")

        driver = gdal.GetDriverByName('GTiff')
        rows, cols = labels.shape
        if labels.dtype != np.uint8:
            labels = labels.astype(np.uint8)

        creation_options = ['PHOTOMETRIC=PALETTE']
        out_ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Byte, creation_options)
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(proj)

        band = out_ds.GetRasterBand(1)
        if color_table is not None:
            band.SetColorTable(color_table)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
        band.WriteArray(labels)
        band.FlushCache()
        out_ds = None

    def _get_output_path(self, input_path):
        """Genera la ruta de salida anadiendo el sufijo de armonizacion."""
        base, ext = os.path.splitext(input_path)
        return f"{base}{self.out_suffix}{ext}"