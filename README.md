# Quantum ESPRESSO Interface Builder 🛠️

A desktop GUI tool for building material interfaces and generating input files for Quantum ESPRESSO (QE).


## Overview ✨

This tool streamlines the process of creating interface models between two different crystalline materials. Given two crystal structure files (e.g., in CIF format), it allows users to:
1.  Generate slabs with specific surface orientations (Miller indices).
2.  Find a common supercell for the two slabs that minimizes lattice mismatch, using either a simple diagonal search or a more robust full-matrix search algorithm.
3.  Stack the matched slabs to form a final interface structure.
4.  Visualize the generated structures in a real-time 3D viewer.
5.  Export ready-to-use input files (`.in`) for `relax`, `vc-relax`, and `scf` calculations in Quantum ESPRESSO.

## Key Features 🚀

* **Intuitive GUI**: All parameters are accessible through a user-friendly interface built with PySide6 (Qt).
* **Advanced Lattice Matching**:
    * **Diagonal Mode**: Searches for simple diagonal supercells.
    * **Full Matrix Mode**: Employs a 2x2 integer matrix search to find non-trivial supercell matches, considering rotations and mirror operations.
    * **Auto-Retry**: If a match isn't found, the tool can automatically relax tolerances and try again.
* **Integrated 3D Viewer**: Real-time visualization of the final interface and individual matched slabs powered by PyVista.
* **Automated QE Input Generation**:
    * Creates input files for both the interface and the individual slabs.
    * Automatically maps pseudopotentials from a specified directory.
    * Calculates a suggested k-point mesh.
    * Generates a simple `run_all.sh` script to execute the QE calculations sequentially.
* **Parameter Management**: Save and load all GUI parameters to a JSON file for reproducibility.

## Installation 📦

1.  **Prerequisites**: Ensure you have Python 3.8+ installed.
2.  **Install Dependencies**: Open your terminal or command prompt and run the following command:
    ```bash
    pip install PySide6 pyvista pyvistaqt ase numpy
    ```

## Usage 📖

1.  **Launch the Application**:
    ```bash
    python QE_interface_builder.py
    ```
2.  **Load Structures (Inputs Tab)**:
    * Click "Browse A" and "Browse B" to select your two CIF files.
    * Enter the Miller indices (`hkl`) for the desired surface of each material.
    * Set the number of layers for each slab and the vacuum/gap distances.
3.  **Configure Matching (Advanced Tab)**:
    * Choose a `Matching mode`. "Full (2x2 matrices)" is recommended for difficult cases.
    * Adjust tolerances for angle and mismatch if needed.
4.  **Build and Preview**:
    * Click the **"Build & Preview"** button.
    * The log panel will show the results of the matching search.
    * Use the dropdown menu above the 3D view to switch between viewing the final `Interface` and the `Matched Slab A` or `Matched Slab B`.
5.  **Set QE Parameters (QE Tab)**:
    * Specify the path to your pseudopotentials directory.
    * Adjust calculation parameters like energy cutoffs, smearing, and k-points.
6.  **Export Files**:
    * Click the **"Export QE Inputs…"** button.
    * Select a directory where the files will be saved.
    * The tool will create subdirectories (`interface`, `slab_A`, `slab_B`) containing the QE input files, CIF files, and a shell script.

## Output Files 📁

Upon exporting, the following structure will be created in your chosen output directory:

```
output_directory/
├── interface/
│   ├── ..._IFACE_relax.in
│   ├── ..._IFACE_scf.in
│   └── ..._IFACE.cif
├── slab_A/
│   ├── ..._A_relax.in
│   ├── ..._A_scf.in
│   └── ..._A.cif
├── slab_B/
│   ├── ..._B_relax.in
│   ├── ..._B_scf.in
│   └── ..._B.cif
└── run_all.sh
```
