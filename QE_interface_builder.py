#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Desktop GUI (native window) for building metal/oxide interfaces with ASE and
exporting Quantum ESPRESSO inputs. Uses PySide6 (Qt) + PyVistaQt (VTK) for 3D.

Install deps (in your env):
  pip install PySide6 pyvista pyvistaqt ase numpy

Run:
  python interface_builder_desktop_v2.py

Notes:
- Preview shows Interface / Matched Slab A / Matched Slab B (switch by combo box).
- If matching is hard, set Mode to "Full (2x2 integer matrices)" and relax tolerances.
- Pseudopotentials are auto-mapped from the folder to elements in the cell.
- QE inputs (relax→scf) are written under selected output directory.
- MODIFIED v23 (Final): Changed Advanced tab to a single-column layout for better resizing.
"""
from __future__ import annotations

import sys
import os
import io
import math
import re
import json
from pathlib import Path
from itertools import product
from typing import Optional, Tuple, List, Dict, Iterable

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.build import surface, stack, make_supercell
from ase.constraints import FixAtoms
from ase.geometry import get_layers
from ase.data import atomic_masses, atomic_numbers, covalent_radii

try:
    from ase.data.colors import jmol_colors
except Exception:
    try:
        from ase.visualize.colors import jmol_colors
    except Exception:
        jmol_colors = np.full((200, 3), 0.6)

# Qt / 3D
from PySide6 import QtCore, QtWidgets, QtGui
from pyvistaqt import QtInteractor
import pyvista as pv

# ------------------------- Math / lattice helpers -------------------------

def fold_angle_deg(ang: float) -> float:
    a = ang % 180.0
    return a if a <= 90.0 else 180.0 - a

def lengths_and_angle_2d(cell_like) -> Tuple[float, float, float]:
    M = np.array(cell_like)
    a, b = np.array(M[0][:2]), np.array(M[1][:2])
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if la < 1e-12 or lb < 1e-12: return la, lb, 90.0
    cosang = float(np.clip(np.dot(a, b) / (la * lb), -1.0, 1.0))
    ang = math.degrees(math.acos(cosang))
    return la, lb, fold_angle_deg(ang)

def area_2d(cell_like) -> float:
    M = np.array(cell_like)
    return abs(M[0,0] * M[1,1] - M[0,1] * M[1,0])

def kmesh_from_cell(cell, kspacing: float = 0.25, kz_min: int = 1, kz_fixed: Optional[int] = None) -> Tuple[int, int, int]:
    M = np.array(cell)
    a,b,c = np.linalg.norm(M[0]), np.linalg.norm(M[1]), np.linalg.norm(M[2])
    twopi = 2.0 * math.pi
    def n_for(L): return max(1, int(round(twopi / max(kspacing * L, 1e-8))))
    kx, ky = n_for(a), n_for(b)
    kz = int(kz_fixed) if kz_fixed not in (None, 0) else max(int(kz_min), n_for(c))
    return (kx, ky, kz)

# ------------------------- ASE helpers -------------------------

def parse_miller(vals: List[int]) -> Tuple[int, int, int]:
    if len(vals) == 3: return int(vals[0]), int(vals[1]), int(vals[2])
    elif len(vals) == 4: h, k, i, l = [int(x) for x in vals]; return h, k, l
    else: raise ValueError("hkl must be 3 numbers or 4 for hex (h k i l)")

def unique_layers_by_z(atoms: Atoms, tol: float = 0.5) -> List[List[int]]:
    idx_sorted = np.argsort(atoms.positions[:, 2])
    zs = atoms.positions[idx_sorted, 2]
    layers, current, z0 = [], [int(idx_sorted[0])], zs[0]
    for j in range(1, len(idx_sorted)):
        if abs(zs[j] - z0) <= tol: current.append(int(idx_sorted[j]))
        else: layers.append(current); current, z0 = [int(idx_sorted[j])], zs[j]
    layers.append(current)
    return layers

def choose_fixed_indices_for_slab(atoms: Atoms, n_bottom: int = 0, n_top: int = 0, tol: float = 0.5) -> List[int]:
    layers = unique_layers_by_z(atoms, tol=tol)
    fix = []
    if n_bottom > 0:
        for lay in layers[:n_bottom]: fix.extend(lay)
    if n_top > 0:
        for lay in layers[-n_top:]: fix.extend(lay)
    return sorted(set(fix))

def find_pseudos(pseudo_dir: Path, elements: List[str]) -> Dict[str, str]:
    if not pseudo_dir.is_dir(): raise FileNotFoundError(f"Pseudo dir not found: {pseudo_dir}")
    files = [p.name for p in pseudo_dir.glob("*.UPF")] + [p.name for p in pseudo_dir.glob("*.upf")]
    mapping: Dict[str, str] = {}
    for el in sorted(set(elements)):
        pat1, pat2 = re.compile(rf"^(?:{el})[._-].*\.u?pf$", re.IGNORECASE), re.compile(rf"(^|[^A-Za-z]){el}([^A-Za-z]|$)", re.IGNORECASE)
        matched = next((f for f in files if pat1.match(f)), None)
        if matched is None: matched = next((f for f in files if pat2.search(f)), None)
        if matched is None and files: matched = files[0]
        if matched is None: raise RuntimeError(f"No UPF files in {pseudo_dir}")
        mapping[el] = matched
    return mapping

def collect_elements(atoms_list: List[Atoms]) -> List[str]:
    return sorted(set(el for at in atoms_list for el in at.get_chemical_symbols()))

def default_spin_mags(elements: List[str]) -> Dict[str, float]:
    presets = {"Fe": 0.6, "Co": 0.5, "Ni": 0.3, "Cr": 0.6, "Mn": 0.6, "V": 0.2}
    return {el: presets[el] for el in elements if el in presets}

def make_slab_from_cif(path: Path, hkl: Tuple[int, int, int], layers: int, vacuum: float) -> Atoms:
    atoms = read(str(path))
    slab = surface(atoms, hkl, layers, vacuum=vacuum)
    slab.center(axis=2); slab.pbc = True
    return slab

def trim_slab_by_thickness(slab: Atoms, max_thick: Optional[float], keep: str) -> Atoms:
    if not max_thick or max_thick <= 0.0: return slab
    at = slab.copy()
    z = at.positions[:, 2]
    tags, _ = get_layers(at, (0, 0, 1))
    uniq = sorted(set(tags))
    layer_z = sorted([(L, float(np.mean(z[tags == L]))) for L in uniq], key=lambda t: t[1], reverse=(keep == 'top'))
    kept = []
    for L, _ in layer_z:
        kept.append(L)
        zz = z[np.isin(tags, kept)]
        span = float(zz.max() - zz.min()) if len(zz) > 0 else 0.0
        if span > max_thick: kept.pop(); break
    if len(kept) < 2 and len(layer_z) >= 2: kept = [layer_z[0][0], layer_z[1][0]]
    at = at[np.isin(tags, kept)]; at.center(axis=2)
    return at

# --------- rotations / supercells ---------
def rotate_cell_xy(cell: np.ndarray, theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    new = cell.copy().astype(float)
    new[0, :2] = (R @ new[0, :2].T).T; new[1, :2] = (R @ new[1, :2].T).T
    return new
def diagonal_supercell(atoms: Atoms, m1: int, m2: int) -> Atoms:
    return make_supercell(atoms, [[m1, 0, 0], [0, m2, 0], [0, 0, 1]])
def set_inplane_to_reference(atoms: Atoms, ref_cell_xy: np.ndarray, mode: str = 'iso'):
    old_positions, old_cell = atoms.positions.copy(), atoms.cell.array.copy()
    if mode == 'none': return
    if mode == 'iso':
        la_old, lb_old, _ = lengths_and_angle_2d(old_cell)
        la_ref, lb_ref, _ = lengths_and_angle_2d(ref_cell_xy)
        s = 0.5 * ((la_ref / la_old) + (lb_ref / lb_old)) if la_old > 1e-8 and lb_old > 1e-8 else 1.0
        temp_cell = old_cell.copy(); temp_cell[0, :2] *= s; temp_cell[1, :2] *= s
        frac = (np.linalg.inv(old_cell) @ old_positions.T).T
        new_pos_scaled = (temp_cell @ frac.T).T; new_pos_scaled[:, 2] = old_positions[:, 2]
        final_cell = temp_cell.copy(); final_cell[0, :2], final_cell[1, :2] = ref_cell_xy[0, :2], ref_cell_xy[1, :2]
        frac2 = (np.linalg.inv(temp_cell) @ new_pos_scaled.T).T
        final_pos = (final_cell @ frac2.T).T; final_pos[:, 2] = old_positions[:, 2]
        atoms.set_cell(final_cell, scale_atoms=False); atoms.set_positions(final_pos)
        return
    if mode == 'aniso':
        new_cell = old_cell.copy(); new_cell[0, :2], new_cell[1, :2] = ref_cell_xy[0, :2], ref_cell_xy[1, :2]
        frac = (np.linalg.inv(old_cell) @ old_positions.T).T
        new_pos = (new_cell @ frac.T).T; new_pos[:, 2] = old_positions[:, 2]
        atoms.set_cell(new_cell, scale_atoms=False); atoms.set_positions(new_pos)
        return
    raise ValueError(f"unknown strain mode: {mode}")
def stack_A_on_B(sA: Atoms, sB: Atoms, gap: float = 2.0, c_pad: float = 0.8) -> Atoms:
    A, B = sA.copy(), sB.copy()
    shift = A.positions[:, 2].max() - B.positions[:, 2].min() + gap
    B.positions[:, 2] += shift
    zlen = B.positions[:, 2].max() - A.positions[:, 2].min() + max(c_pad, 0.0)
    cell = A.cell.array.copy()
    cell[2, :] = [0.0, 0.0, zlen]
    A.set_cell(cell, scale_atoms=False); B.set_cell(cell, scale_atoms=False)
    return A + B

# ---- full-matrix search ----
def gen_int_mats(max_entry: int, det_max: int) -> Iterable[np.ndarray]:
    rng = range(-max_entry, max_entry + 1)
    for a, b, c, d in product(rng, rng, rng, rng):
        M = np.array([[a, b], [c, d]], dtype=int)
        det = int(round(a * d - b * c))
        if det != 0 and abs(det) <= det_max: yield M
def apply_2d_mat(cell2x2: np.ndarray, M: np.ndarray) -> np.ndarray:
    A = cell2x2[:2, :2].copy()
    return np.vstack([M[0, 0] * A[0] + M[0, 1] * A[1], M[1, 0] * A[0] + M[1, 1] * A[1]])
def match_quality(A2: np.ndarray, B2: np.ndarray, angle_tol: float, len_tol_pct: float) -> Tuple[bool, float, float, float]:
    la1, lb1, ang1 = lengths_and_angle_2d(A2); la2, lb2, ang2 = lengths_and_angle_2d(B2)
    L1 = sorted([la1, lb1], reverse=True); L2 = sorted([la2, lb2], reverse=True)
    mis1 = abs(L1[0] - L2[0]) / max(L1[0], 1e-9) * 100.0
    mis2 = abs(L1[1] - L2[1]) / max(L1[1], 1e-9) * 100.0
    dangle = abs(ang1 - ang2)
    return (max(mis1, mis2) <= len_tol_pct) and (dangle <= angle_tol), mis1, mis2, dangle
def search_full_integer_supercells(
    slabA: Atoms, slabB: Atoms, *, max_entry: int = 3, det_max: int = 12, angle_step: float = 1.0,
    angle_tol: float = 3.0, max_mismatch: float = 4.0, area_limit: Optional[float] = 2000.0,
    prefer_small: bool = True, try_mirror: bool = True, swap_roles: bool = True
):
    Axy, Bxy_orig = slabA.cell.array.copy()[:2, :2], slabB.cell.array.copy()[:2, :2]
    mirrors = [np.eye(2), np.diag([-1, 1])] if try_mirror else [np.eye(2)]
    best, mats = None, list(gen_int_mats(max_entry=max_entry, det_max=det_max))
    for ref_is_A in ([True, False] if swap_roles else [True]):
        Aref, Bref0 = (Axy, Bxy_orig) if ref_is_A else (Bxy_orig, Axy)
        for U in mats:
            AU = apply_2d_mat(Aref, U)
            if area_limit is not None and area_2d(AU) > area_limit: continue
            for V in mats:
                BV0 = apply_2d_mat(Bref0, V)
                for Mir in mirrors:
                    BVm = (Mir @ BV0.T).T
                    for theta in np.arange(0.0, 180.0 + 1e-9, angle_step):
                        Mrot = np.array([[math.cos(math.radians(theta)), -math.sin(math.radians(theta))],
                                         [math.sin(math.radians(theta)),  math.cos(math.radians(theta))]])
                        BVr = (Mrot @ BVm.T).T
                        ok, mis1, mis2, dangle = match_quality(AU, BVr, angle_tol=angle_tol, len_tol_pct=max_mismatch)
                        if not ok: continue
                        mA, mB = abs(int(round(np.linalg.det(U)))), abs(int(round(np.linalg.det(V))))
                        size_pen, imb = 0.5 * (mA + mB), abs(math.log(max(mA, 1) / max(mB, 1)))
                        score = 0.5 * (mis1 + mis2) + 0.3 * dangle + (0.6 if prefer_small else 0.3) * size_pen + (0.3 if prefer_small else 0.2) * imb
                        cand = dict(ref_is_A=ref_is_A, U=U, V=V, theta=theta, mis1=mis1, mis2=mis2, dangle=dangle,
                                    mA=mA, mB=mB, areaA=area_2d(AU), score=score)
                        if (best is None) or (score < best['score']): best = cand
    if best is None: raise RuntimeError("No match by full-matrix search. Relax tolerances.")
    return best

# ------------------------- QE writer -------------------------
def write_qe_input_manual(atoms: Atoms, filename: str, kpts: Tuple[int, int, int], params: Dict):
    calc_params = params.copy()
    pseudo_dir, pseudos = calc_params.pop('pseudo_dir'), calc_params.pop('pseudos')
    calculation, prefix = calc_params.pop('calculation'), calc_params.pop('prefix')
    spin_mags = calc_params.pop('spin_mags', None)
    symbols, unique_symbols = atoms.get_chemical_symbols(), sorted(set(atoms.get_chemical_symbols()))
    nat, ntyp = len(atoms), len(unique_symbols)
    fixed_indices = []
    if len(atoms.constraints) > 0:
        for constr in atoms.constraints:
            if isinstance(constr, FixAtoms): fixed_indices.extend(constr.get_indices())
    fixed = set(fixed_indices)
    ifpos = [(0, 0, 0) if i in fixed else (1, 1, 1) for i in range(nat)]
    lines = [f"&CONTROL", f"  calculation = '{calculation}',", f"  prefix = '{prefix}',",
             f"  pseudo_dir = '{str(pseudo_dir)}',", f"  outdir = './tmp_{prefix}',",
             f"  tstress = .true.,", f"  tprnfor = .true.,", f"  restart_mode = 'from_scratch',", f"/",
             f"&SYSTEM", f"  ibrav = 0,", f"  nat = {nat},", f"  ntyp = {ntyp},",
             f"  ecutwfc = {calc_params.get('ecutwfc', 50.0)},", f"  ecutrho = {calc_params.get('ecutrho', 400.0)},",
             f"  occupations = 'smearing',", f"  smearing = '{calc_params.get('smearing', 'mv')}',",
             f"  degauss = {calc_params.get('degauss', 0.02)},"]
    if spin_mags:
        lines.append(f"  nspin = 2,")
        for i, symbol in enumerate(unique_symbols, start=1):
            mag_val = spin_mags.get(symbol, 0.0)
            lines.append(f"  starting_magnetization({i}) = {mag_val:.4f},")
    lines.append(f"/")
    lines.append(f"&ELECTRONS")
    if 'mixing_mode' in calc_params and calc_params['mixing_mode']:
        lines.append(f"  mixing_mode = '{calc_params['mixing_mode']}',")
    lines.append(f"  mixing_beta = {calc_params.get('mixing_beta', 0.3):.2f},")
    default_conv_thr = '1.0e-8' if calculation == 'scf' else '1.0e-6'
    lines.append(f"  conv_thr = {calc_params.get('conv_thr', default_conv_thr)},")
    lines.append(f"/")
    if calculation in ('relax', 'vc-relax'): lines.extend([f"&IONS", f"  ion_dynamics = 'bfgs',", f"/"])
    if calculation == 'vc-relax': lines.extend([f"&CELL", f"  cell_dynamics = 'bfgs',", f"  cell_dofree = '2Dxy',", f"/"])
    lines.append(f"ATOMIC_SPECIES")
    masses = {s: atomic_masses[atomic_numbers[s]] for s in unique_symbols}
    for s in unique_symbols: lines.append(f"  {s:<3} {masses[s]:.4f}  {pseudos[s]}")
    lines.append(f"CELL_PARAMETERS angstrom")
    for vec in atoms.cell: lines.append(f"  {vec[0]:16.10f} {vec[1]:16.10f} {vec[2]:16.10f}")
    lines.append(f"ATOMIC_POSITIONS {{angstrom}}")
    for i, (symbol, pos) in enumerate(zip(symbols, atoms.positions)):
        fx, fy, fz = ifpos[i]
        lines.append(f"  {symbol:<3} {pos[0]:16.10f} {pos[1]:16.10f} {pos[2]:16.10f}   {fx} {fy} {fz}")
    lines.append(f"K_POINTS {{automatic}}")
    lines.append(f"  {kpts[0]} {kpts[1]} {kpts[2]}   0 0 0")
    with open(filename, 'w', encoding='utf-8') as f: f.write("\n".join(lines) + "\n")

# ------------------------- 3D visualization -------------------------
def add_atoms_actor(plotter: QtInteractor, atoms: Atoms):
    plotter.clear()
    if atoms is None or len(atoms) == 0:
        if atoms is not None and not np.all(atoms.cell.lengths() == 0):
            cell, origin = atoms.cell.array, np.zeros(3)
            a, b, c = cell[0], cell[1], cell[2]
            box_points = np.array([origin, a, a + b, b, origin + c, a + c, a + b + c, b + c])
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i, j in edges: plotter.add_lines(np.vstack([box_points[i], box_points[j]]), width=1)
        plotter.reset_camera()
        return

    atoms_to_draw = atoms.copy()
    atoms_to_draw.wrap()

    positions, symbols, n = atoms_to_draw.get_positions(), atoms_to_draw.get_chemical_symbols(), len(atoms_to_draw)
    fixed_mask = np.zeros(n, dtype=bool)
    if len(atoms_to_draw.constraints) > 0:
        for constr in atoms_to_draw.constraints:
            if isinstance(constr, FixAtoms):
                fixed_mask[constr.get_indices()] = True

    for i, (symbol, pos) in enumerate(zip(symbols, positions)):
        radius = covalent_radii[atomic_numbers.get(symbol, 0)] * 0.6
        z = atomic_numbers.get(symbol)
        color = jmol_colors[z] if z is not None and z < len(jmol_colors) else (0.6, 0.6, 0.6)
        if fixed_mask[i]: color = (1.0, 0.9, 0.0)
        sphere = pv.Sphere(radius=radius, center=pos, theta_resolution=16, phi_resolution=16)
        plotter.add_mesh(sphere, color=color, specular=0.5, specular_power=10, smooth_shading=True)

    cell = atoms_to_draw.cell
    if not np.all(cell.lengths() == 0):
        origin = np.zeros(3)
        a, b, c = cell[0], cell[1], cell[2]
        box_points = np.array([origin, a, a + b, b, origin + c, a + c, a + b + c, b + c])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges: plotter.add_lines(np.vstack([box_points[i], box_points[j]]), width=2, color='gray')
            
    plotter.reset_camera()

# ------------------------- Qt Main Window -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface Builder for Quantum ESPRESSO")
        self.resize(1200, 900)
        self.slabA, self.slabB, self.iface, self.sA, self.sB = None, None, None, None, None
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False) # MODIFIED
        main_layout.addWidget(self.splitter)
        self.ctrl_tabs = QtWidgets.QTabWidget()
        self.splitter.addWidget(self.ctrl_tabs)
        right_panel = QtWidgets.QWidget(); right_layout = QtWidgets.QHBoxLayout(right_panel)
        right_layout.setContentsMargins(0,0,0,0)
        view_log_widget = QtWidgets.QWidget(); view_log_layout = QtWidgets.QVBoxLayout(view_log_widget)
        self.plotter = QtInteractor(view_log_widget); 
        self.plotter.add_axes()
        view_log_layout.addWidget(self.plotter.interactor)
        info_area = QtWidgets.QWidget(); info_layout = QtWidgets.QHBoxLayout(info_area)
        info_layout.setContentsMargins(0, 5, 0, 0)
        self.model_combo = QtWidgets.QComboBox(); self.model_combo.addItems(["Interface", "Matched Slab A", "Matched Slab B"])
        self.model_combo.currentTextChanged.connect(self.update_view)
        self.info_ntyp_label = QtWidgets.QLabel("Ntyp: -")
        info_layout.addWidget(self.model_combo); info_layout.addWidget(self.info_ntyp_label); info_layout.addStretch()
        view_log_layout.addWidget(info_area)
        self.info = QtWidgets.QPlainTextEdit(); self.info.setReadOnly(True)
        view_log_layout.addWidget(self.info); right_layout.addWidget(view_log_widget)
        button_widget = QtWidgets.QWidget(); button_layout = QtWidgets.QVBoxLayout(button_widget)
        self.btn_load_params = QtWidgets.QPushButton("Load Params"); self.btn_save_params = QtWidgets.QPushButton("Save Params")
        self.btn_build = QtWidgets.QPushButton("Build & Preview"); self.btn_export = QtWidgets.QPushButton("Export QE Inputs…")
        button_layout.addWidget(self.btn_load_params); button_layout.addWidget(self.btn_save_params)
        button_layout.addWidget(self.btn_build); button_layout.addWidget(self.btn_export); button_layout.addStretch()
        right_layout.addWidget(button_widget)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([400, 800])
        
        self.build_input_tab()
        self.build_advanced_tab()
        self.build_qe_tab()
        
        self.btn_build.clicked.connect(self.start_build_process); self.btn_export.clicked.connect(self.export_qe)
        self.btn_save_params.clicked.connect(self.save_parameters); self.btn_load_params.clicked.connect(self.load_parameters)

    def build_input_tab(self):
        t_in = QtWidgets.QWidget(); f_in = QtWidgets.QFormLayout(t_in)
        self.pathA = QtWidgets.QLineEdit(); btnA = QtWidgets.QPushButton("Browse A"); self.pathB = QtWidgets.QLineEdit(); btnB = QtWidgets.QPushButton("Browse B")
        btnA.clicked.connect(lambda: self.pick_file(self.pathA)); btnB.clicked.connect(lambda: self.pick_file(self.pathB))
        rowA = QtWidgets.QHBoxLayout(); rowA.addWidget(self.pathA); rowA.addWidget(btnA)
        rowB = QtWidgets.QHBoxLayout(); rowB.addWidget(self.pathB); rowB.addWidget(btnB)
        boxA = QtWidgets.QWidget(); boxA.setLayout(rowA); boxB = QtWidgets.QWidget(); boxB.setLayout(rowB)
        f_in.addRow("CIF A", boxA); f_in.addRow("CIF B", boxB)
        self.hkl1 = QtWidgets.QLineEdit("1 1 0"); self.hkl2 = QtWidgets.QLineEdit("0 0 0 1")
        f_in.addRow("hkl A (3 or 4)", self.hkl1); f_in.addRow("hkl B (3 or 4)", self.hkl2)
        self.layers1 = QtWidgets.QSpinBox(); self.layers1.setRange(2, 200); self.layers1.setValue(8)
        self.layers2 = QtWidgets.QSpinBox(); self.layers2.setRange(2, 200); self.layers2.setValue(8)
        f_in.addRow("layers A", self.layers1); f_in.addRow("layers B", self.layers2)
        self.vacuum = QtWidgets.QDoubleSpinBox(); self.vacuum.setRange(0, 100); self.vacuum.setValue(15.0); self.vacuum.setDecimals(1)
        self.gap = QtWidgets.QDoubleSpinBox(); self.gap.setRange(0.5, 10); self.gap.setValue(2.0); self.gap.setDecimals(2)
        f_in.addRow("vacuum (Å)", self.vacuum); f_in.addRow("gap (Å)", self.gap)
        self.maxstrain = QtWidgets.QDoubleSpinBox(); self.maxstrain.setRange(0.0, 0.2); self.maxstrain.setSingleStep(0.01); self.maxstrain.setValue(0.04)
        f_in.addRow("stack maxstrain", self.maxstrain)
        self.fix_bot = QtWidgets.QSpinBox(); self.fix_bot.setRange(0, 20); self.fix_bot.setValue(2)
        self.fix_top = QtWidgets.QSpinBox(); self.fix_top.setRange(0, 20); self.fix_top.setValue(2)
        f_in.addRow("Fix bottom layers (A)", self.fix_bot); f_in.addRow("Fix top layers (B)", self.fix_top)
        self.vcrelax = QtWidgets.QCheckBox("vc-relax (2Dxy)"); f_in.addRow(self.vcrelax)
        self.ctrl_tabs.addTab(t_in, "Inputs")

    def build_advanced_tab(self):
        t_adv = QtWidgets.QWidget()
        f_adv = QtWidgets.QFormLayout(t_adv)
        def add_subtitle(text):
            label = QtWidgets.QLabel(f"<b>{text}</b>")
            f_adv.addRow(label)

        self.mode = QtWidgets.QComboBox(); self.mode.addItems(["Diagonal", "Full (2x2 matrices)", "stack (ASE)"])
        f_adv.addRow("Matching mode", self.mode)
        
        add_subtitle("Common Search Parameters")
        self.angle_step = QtWidgets.QDoubleSpinBox(); self.angle_step.setRange(0.1, 10.0); self.angle_step.setSingleStep(0.1); self.angle_step.setValue(1.0)
        f_adv.addRow("angle_step (°)", self.angle_step)
        self.angle_tol = QtWidgets.QDoubleSpinBox(); self.angle_tol.setRange(0.5, 180.0); self.angle_tol.setSingleStep(0.5); self.angle_tol.setValue(3.0)
        f_adv.addRow("angle_tol (°)", self.angle_tol)
        self.max_mismatch = QtWidgets.QDoubleSpinBox(); self.max_mismatch.setRange(0.5, 15.0); self.max_mismatch.setSingleStep(0.5); self.max_mismatch.setValue(4.0)
        f_adv.addRow("max_mismatch (%)", self.max_mismatch)
        self.strain_mode = QtWidgets.QComboBox(); self.strain_mode.addItems(["iso","aniso","none"])
        self.prefer_small = QtWidgets.QCheckBox("Prefer small supercells"); self.prefer_small.setChecked(True)
        hbox1 = QtWidgets.QHBoxLayout(); hbox1.addWidget(self.strain_mode); hbox1.addWidget(self.prefer_small)
        f_adv.addRow("Strain & Scoring", hbox1)

        add_subtitle("Diagonal Search")
        self.mmax = QtWidgets.QSpinBox(); self.mmax.setRange(2, 30); self.mmax.setValue(10)
        f_adv.addRow("mmax", self.mmax)
        self.max_mAprod = QtWidgets.QSpinBox(); self.max_mAprod.setRange(0, 200); self.max_mAprod.setValue(0)
        f_adv.addRow("max_mA* (0=off)", self.max_mAprod)
        self.max_mBprod = QtWidgets.QSpinBox(); self.max_mBprod.setRange(0, 200); self.max_mBprod.setValue(0)
        f_adv.addRow("max_mB* (0=off)", self.max_mBprod)
        self.max_Aarea = QtWidgets.QDoubleSpinBox(); self.max_Aarea.setRange(0.0, 10000.0); self.max_Aarea.setDecimals(1); self.max_Aarea.setValue(0.0)
        f_adv.addRow("max_Area Å² (0=off)", self.max_Aarea)

        add_subtitle("Full Matrix Search")
        self.max_entry = QtWidgets.QSpinBox(); self.max_entry.setRange(1, 8); self.max_entry.setValue(3)
        self.det_max = QtWidgets.QSpinBox(); self.det_max.setRange(1, 40); self.det_max.setValue(12)
        hbox2 = QtWidgets.QHBoxLayout(); hbox2.addWidget(self.max_entry); hbox2.addWidget(QtWidgets.QLabel("det_max:")); hbox2.addWidget(self.det_max)
        f_adv.addRow("max_entry / det_max", hbox2)
        self.area_limit = QtWidgets.QDoubleSpinBox(); self.area_limit.setRange(0.0, 10000.0); self.area_limit.setDecimals(1); self.area_limit.setValue(2000.0)
        f_adv.addRow("area_limit Å² (0=off)", self.area_limit)
        self.try_mirror = QtWidgets.QCheckBox("Try mirror (B)"); self.try_mirror.setChecked(True)
        self.swap_roles = QtWidgets.QCheckBox("Swap A/B roles"); self.swap_roles.setChecked(True)
        hbox3 = QtWidgets.QHBoxLayout(); hbox3.addWidget(self.try_mirror); hbox3.addWidget(self.swap_roles)
        f_adv.addRow("Options", hbox3)
        
        add_subtitle("Slab Geometry")
        self.max_thickA = QtWidgets.QDoubleSpinBox(); self.max_thickA.setRange(0.0, 200.0); self.max_thickA.setDecimals(1); self.max_thickA.setValue(0.0)
        self.max_thickB = QtWidgets.QDoubleSpinBox(); self.max_thickB.setRange(0.0, 200.0); self.max_thickB.setDecimals(1); self.max_thickB.setValue(0.0)
        f_adv.addRow("max_thickA (Å)", self.max_thickA)
        f_adv.addRow("max_thickB (Å)", self.max_thickB)
        self.c_pad = QtWidgets.QDoubleSpinBox(); self.c_pad.setRange(0.0, 40.0); self.c_pad.setDecimals(1); self.c_pad.setValue(10.0)
        f_adv.addRow("c_pad (Å)", self.c_pad)
        self.ctrl_tabs.addTab(t_adv, "Advanced")

    def build_qe_tab(self):
        t_qe = QtWidgets.QWidget(); f_qe = QtWidgets.QFormLayout(t_qe)
        def add_subtitle(text):
            label = QtWidgets.QLabel(f"<b>{text}</b>")
            f_qe.addRow(label)
        
        add_subtitle("&CONTROL")
        self.pseudo_dir = QtWidgets.QLineEdit(); btnP = QtWidgets.QPushButton("Browse")
        btnP.clicked.connect(lambda: self.pick_dir(self.pseudo_dir))
        rowP = QtWidgets.QHBoxLayout(); rowP.addWidget(self.pseudo_dir); rowP.addWidget(btnP)
        wP = QtWidgets.QWidget(); wP.setLayout(rowP); f_qe.addRow("pseudo_dir", wP)
        self.workdir = QtWidgets.QLineEdit(); f_qe.addRow("workdir", self.workdir)
        
        add_subtitle("&SYSTEM")
        self.ecutwfc = QtWidgets.QDoubleSpinBox(); self.ecutwfc.setRange(10, 200); self.ecutwfc.setValue(45)
        self.ecutrho = QtWidgets.QDoubleSpinBox(); self.ecutrho.setRange(40, 2000); self.ecutrho.setValue(360)
        f_qe.addRow("ecutwfc (Ry)", self.ecutwfc); f_qe.addRow("ecutrho (Ry)", self.ecutrho)
        self.smearing = QtWidgets.QComboBox(); self.smearing.addItems(["mv", "mp", "gauss", "fd"])
        self.degauss = QtWidgets.QDoubleSpinBox(); self.degauss.setRange(0.001, 0.2); self.degauss.setSingleStep(0.005); self.degauss.setValue(0.03)
        f_qe.addRow("smearing", self.smearing); f_qe.addRow("degauss (Ry)", self.degauss)
        self.enable_spin = QtWidgets.QCheckBox("Enable Spin Polarization (auto-presets)"); self.enable_spin.setChecked(False)
        f_qe.addRow(self.enable_spin)
        
        add_subtitle("&ELECTRONS")
        self.mixing_mode = QtWidgets.QComboBox(); self.mixing_mode.addItems(["plain", "TF", "local-TF"])
        f_qe.addRow("mixing_mode", self.mixing_mode)
        self.mixing_beta = QtWidgets.QDoubleSpinBox(); self.mixing_beta.setRange(0.01, 1.0); self.mixing_beta.setDecimals(2); self.mixing_beta.setSingleStep(0.1); self.mixing_beta.setValue(0.3)
        f_qe.addRow("mixing_beta", self.mixing_beta)
        self.conv_thr = QtWidgets.QLineEdit("1.0e-6")
        f_qe.addRow("conv_thr", self.conv_thr)
        
        f_qe.addRow(QtWidgets.QLabel("--- K-Points ---"))
        self.kpoint_mode_combo = QtWidgets.QComboBox(); self.kpoint_mode_combo.addItems(["Automatic (k-spacing)", "Manual"])
        self.kpoint_mode_combo.currentTextChanged.connect(self.update_kpoint_widgets)
        f_qe.addRow("K-Point Mode", self.kpoint_mode_combo)
        self.kspacing_widget = QtWidgets.QWidget()
        kspacing_layout = QtWidgets.QFormLayout(self.kspacing_widget); kspacing_layout.setContentsMargins(0,0,0,0)
        self.kspacing = QtWidgets.QDoubleSpinBox(); self.kspacing.setRange(0.1, 1.5); self.kspacing.setDecimals(2); self.kspacing.setValue(0.25)
        self.kz_min = QtWidgets.QSpinBox(); self.kz_min.setRange(1, 12); self.kz_min.setValue(1)
        self.kz_fixed = QtWidgets.QSpinBox(); self.kz_fixed.setRange(0, 12); self.kz_fixed.setValue(1)
        kspacing_layout.addRow("k-spacing (1/Å)", self.kspacing)
        kspacing_layout.addRow("kz_min", self.kz_min)
        kspacing_layout.addRow("kz_fixed (0=auto)", self.kz_fixed)
        f_qe.addRow(self.kspacing_widget)
        self.kpoint_manual_edit = QtWidgets.QLineEdit("4 4 1")
        f_qe.addRow("Manual K-Points", self.kpoint_manual_edit)
        self.ctrl_tabs.addTab(t_qe, "QE")
        self.update_kpoint_widgets()

    def update_kpoint_widgets(self):
        is_manual = self.kpoint_mode_combo.currentText() == "Manual"
        self.kspacing_widget.setEnabled(not is_manual)
        self.kpoint_manual_edit.setEnabled(is_manual)
    
    def pick_file(self, line: QtWidgets.QLineEdit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CIF", os.getcwd(), "CIF Files (*.cif)")
        if path: line.setText(path)
    def pick_dir(self, line: QtWidgets.QLineEdit):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if path: line.setText(path)
    def _parse_hkl(self, text: str) -> Tuple[int, int, int]:
        return parse_miller([int(x) for x in text.replace(',', ' ').split() if x.strip()])
    def log(self, msg: str):
        self.info.appendPlainText(msg); QtWidgets.QApplication.processEvents()
    def save_parameters(self):
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Parameters", os.getcwd(), "JSON Files (*.json)")
        if not save_path: return
        params = { 'pathA': self.pathA.text(), 'pathB': self.pathB.text(), 'hkl1': self.hkl1.text(), 'hkl2': self.hkl2.text(), 'layers1': self.layers1.value(), 'layers2': self.layers2.value(), 'vacuum': self.vacuum.value(), 'gap': self.gap.value(), 'maxstrain': self.maxstrain.value(), 'fix_bot': self.fix_bot.value(), 'fix_top': self.fix_top.value(), 'vcrelax': self.vcrelax.isChecked(), 'mode': self.mode.currentText(), 'mmax': self.mmax.value(), 'angle_step': self.angle_step.value(), 'angle_tol': self.angle_tol.value(), 'max_mismatch': self.max_mismatch.value(), 'strain_mode': self.strain_mode.currentText(), 'prefer_small': self.prefer_small.isChecked(), 'max_mAprod': self.max_mAprod.value(), 'max_mBprod': self.max_mBprod.value(), 'max_Aarea': self.max_Aarea.value(), 'max_entry': self.max_entry.value(), 'det_max': self.det_max.value(), 'area_limit': self.area_limit.value(), 'try_mirror': self.try_mirror.isChecked(), 'swap_roles': self.swap_roles.isChecked(), 'max_thickA': self.max_thickA.value(), 'max_thickB': self.max_thickB.value(), 'c_pad': self.c_pad.value(), 'pseudo_dir': self.pseudo_dir.text(), 'workdir': self.workdir.text(), 'ecutwfc': self.ecutwfc.value(), 'ecutrho': self.ecutrho.value(), 'kspacing': self.kspacing.value(), 'kz_min': self.kz_min.value(), 'kz_fixed': self.kz_fixed.value(), 'degauss': self.degauss.value(), 'smearing': self.smearing.currentText(), 'enable_spin': self.enable_spin.isChecked(), 'mixing_mode': self.mixing_mode.currentText(), 'mixing_beta': self.mixing_beta.value(), 'conv_thr': self.conv_thr.text(), 'kpoint_mode': self.kpoint_mode_combo.currentText(), 'kpoint_manual': self.kpoint_manual_edit.text() }
        try:
            with open(save_path, 'w', encoding='utf-8') as f: json.dump(params, f, indent=4)
            self.log(f"Parameters saved to: {save_path}")
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save parameters file.\nError: {e}")
    def load_parameters(self):
        load_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Parameters", os.getcwd(), "JSON Files (*.json)")
        if not load_path: return
        try:
            with open(load_path, 'r', encoding='utf-8') as f: params = json.load(f)
            self.pathA.setText(params.get('pathA', '')); self.pathB.setText(params.get('pathB', '')); self.hkl1.setText(params.get('hkl1', '')); self.hkl2.setText(params.get('hkl2', ''))
            self.layers1.setValue(params.get('layers1', 8)); self.layers2.setValue(params.get('layers2', 8)); self.vacuum.setValue(params.get('vacuum', 15.0)); self.gap.setValue(params.get('gap', 2.0))
            self.maxstrain.setValue(params.get('maxstrain', 0.04)); self.fix_bot.setValue(params.get('fix_bot', 2)); self.fix_top.setValue(params.get('fix_top', 2)); self.vcrelax.setChecked(params.get('vcrelax', False))
            self.mode.setCurrentText(params.get('mode', 'Diagonal')); self.mmax.setValue(params.get('mmax', 10)); self.angle_step.setValue(params.get('angle_step', 1.0)); self.angle_tol.setValue(params.get('angle_tol', 3.0))
            self.max_mismatch.setValue(params.get('max_mismatch', 4.0)); self.strain_mode.setCurrentText(params.get('strain_mode', 'iso')); self.prefer_small.setChecked(params.get('prefer_small', True))
            self.max_mAprod.setValue(params.get('max_mAprod', 0)); self.max_mBprod.setValue(params.get('max_mBprod', 0)); self.max_Aarea.setValue(params.get('max_Aarea', 0.0))
            self.max_entry.setValue(params.get('max_entry', 3)); self.det_max.setValue(params.get('det_max', 12)); self.area_limit.setValue(params.get('area_limit', 2000.0)); self.try_mirror.setChecked(params.get('try_mirror', True))
            self.swap_roles.setChecked(params.get('swap_roles', True)); self.max_thickA.setValue(params.get('max_thickA', 0.0)); self.max_thickB.setValue(params.get('max_thickB', 0.0)); self.c_pad.setValue(params.get('c_pad', 0.8))
            self.pseudo_dir.setText(params.get('pseudo_dir', '')); self.workdir.setText(params.get('workdir', '')); self.ecutwfc.setValue(params.get('ecutwfc', 45.0)); self.ecutrho.setValue(params.get('ecutrho', 360.0))
            self.kspacing.setValue(params.get('kspacing', 0.25)); self.kz_min.setValue(params.get('kz_min', 1)); self.kz_fixed.setValue(params.get('kz_fixed', 0)); self.degauss.setValue(params.get('degauss', 0.03)); self.smearing.setCurrentText(params.get('smearing', 'mv'))
            self.enable_spin.setChecked(params.get('enable_spin', False)); self.mixing_mode.setCurrentText(params.get('mixing_mode', 'plain'))
            self.mixing_beta.setValue(params.get('mixing_beta', 0.3)); self.conv_thr.setText(params.get('conv_thr', '1.0e-6'))
            self.kpoint_mode_combo.setCurrentText(params.get('kpoint_mode', 'Automatic (k-spacing)')); self.kpoint_manual_edit.setText(params.get('kpoint_manual', '4 4 1'))
            self.update_kpoint_widgets()
            self.log(f"Parameters loaded from: {load_path}")
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load or parse parameters file.\nError: {e}")
    def start_build_process(self):
        self.info.clear()
        try: hkl1, hkl2 = self._parse_hkl(self.hkl1.text()), self._parse_hkl(self.hkl2.text())
        except Exception as e: QtWidgets.QMessageBox.critical(self, "HKL error", str(e)); return
        pA, pB = Path(self.pathA.text().strip()), Path(self.pathB.text().strip())
        if not (pA.exists() and pB.exists()): QtWidgets.QMessageBox.warning(self, "Missing file", "Please set both CIF paths."); return
        try:
            self.slabA = make_slab_from_cif(pA, hkl1, self.layers1.value(), self.vacuum.value())
            self.slabB = make_slab_from_cif(pB, hkl2, self.layers2.value(), self.vacuum.value())
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Slab creation error", f"Failed to create initial slabs: {e}"); return
        search_presets = [ {'label': 'User settings'}, {'label': 'Relaxed', 'angle_tol': 5.0, 'max_mismatch': 7.0, 'mmax': 15, 'max_entry': 4, 'det_max': 20, 'area_limit': 3000.0}, {'label': 'Very Relaxed', 'angle_tol': 8.0, 'max_mismatch': 10.0, 'mmax': 20, 'max_entry': 5, 'det_max': 30, 'area_limit': 4000.0} ]
        original_params = { 'angle_tol': self.angle_tol.value(), 'max_mismatch': self.max_mismatch.value(), 'mmax': self.mmax.value(), 'max_entry': self.max_entry.value(), 'det_max': self.det_max.value(), 'area_limit': self.area_limit.value() }
        success = False
        for i, params in enumerate(search_presets):
            self.log(f"\n--- Attempt #{i+1}: Using '{params['label']}' ---")
            if i > 0:
                self.angle_tol.setValue(params['angle_tol']); self.max_mismatch.setValue(params['max_mismatch']); self.mmax.setValue(params['mmax']); self.max_entry.setValue(params['max_entry']); self.det_max.setValue(params['det_max']); self.area_limit.setValue(params['area_limit'])
                self.log(f"Updated params: angle_tol={params['angle_tol']}°, max_mismatch={params['max_mismatch']}%, det_max={params['det_max']}")
            if self.run_build_logic(): self.log(f"\n✅ Success on attempt #{i+1}!"); success = True; break
            else: self.log(f"Attempt #{i+1} failed to find a match. Trying more relaxed parameters...")
        if not success:
            self.log("\n❌ All attempts failed."); QtWidgets.QMessageBox.critical(self, "Matching Failed", "Could not find a matching interface even with relaxed parameters.")
            self.angle_tol.setValue(original_params['angle_tol']); self.max_mismatch.setValue(original_params['max_mismatch']); self.mmax.setValue(original_params['mmax']); self.max_entry.setValue(original_params['max_entry']); self.det_max.setValue(original_params['det_max']); self.area_limit.setValue(original_params['area_limit'])
    def run_build_logic(self) -> bool:
        if self.slabA is None or self.slabB is None: self.log("Error: Base slabs are not available."); return False
        idx_fix_A = choose_fixed_indices_for_slab(self.slabA, n_bottom=self.fix_bot.value(), n_top=0)
        idx_fix_B = choose_fixed_indices_for_slab(self.slabB, n_bottom=0, n_top=self.fix_top.value())
        maskA = np.zeros(len(self.slabA), dtype=bool); maskA[idx_fix_A] = True
        maskB = np.zeros(len(self.slabB), dtype=bool); maskB[idx_fix_B] = True
        slabA_copy, slabB_copy = self.slabA.copy(), self.slabB.copy()
        slabA_copy.set_constraint(FixAtoms(mask=maskA)); slabB_copy.set_constraint(FixAtoms(mask=maskB))
        mode = self.mode.currentText()
        self.iface, self.sA, self.sB = None, None, None
        try:
            if mode.startswith("stack"):
                self.iface = stack(slabA_copy, slabB_copy, axis=2, maxstrain=self.maxstrain.value(), distance=self.gap.value())
            elif mode.startswith("Diagonal"):
                sA0, sB0 = trim_slab_by_thickness(slabA_copy, self.max_thickA.value(), 'top'), trim_slab_by_thickness(slabB_copy, self.max_thickB.value(), 'bottom')
                best = None
                max_mAprod_val, max_mBprod_val, max_Aarea_val = self.max_mAprod.value(), self.max_mBprod.value(), self.max_Aarea.value()
                laA, lbA, angA_rad = lengths_and_angle_2d(sA0.cell)
                areaA0 = laA * lbA * math.sin(math.radians(angA_rad)) if laA > 0 and lbA > 0 else 0.0
                for theta in np.arange(0.0, 180.0 + 1e-9, self.angle_step.value()):
                    laA_cell, lbA_cell, angA = lengths_and_angle_2d(sA0.cell)
                    laB, lbB, angB = lengths_and_angle_2d(rotate_cell_xy(sB0.cell.array, theta))
                    if abs(fold_angle_deg(angA - angB)) > self.angle_tol.value(): continue
                    for mA1, mA2, mB1, mB2 in product(range(1, self.mmax.value()+1), repeat=4):
                        mAprod = mA1*mA2; mBprod = mB1*mB2; areaA = areaA0*mAprod
                        if (max_mAprod_val > 0 and mAprod > max_mAprod_val) or \
                           (max_mBprod_val > 0 and mBprod > max_mBprod_val) or \
                           (max_Aarea_val > 0 and areaA > max_Aarea_val): continue
                        A1, A2 = laA_cell*mA1, lbA_cell*mA2; B1, B2 = laB*mB1, lbB*mB2
                        s = 1.0 if (B1<1e-8 or B2<1e-8) else 0.5*((A1/B1)+(A2/B2))
                        mis1, mis2 = abs(A1-s*B1)/max(A1,1e-12)*100.0, abs(A2-s*B2)/max(A2,1e-12)*100.0
                        if max(mis1, mis2) > self.max_mismatch.value(): continue
                        size_pen, imb = 0.5*(mAprod+mBprod), abs(math.log(max(mAprod,1)/max(mBprod,1)))
                        score = 0.5*(mis1+mis2) + 0.3*abs(fold_angle_deg(angA-angB)) + (0.6 if self.prefer_small.isChecked() else 0.3)*size_pen + (0.3 if self.prefer_small.isChecked() else 0.2)*imb
                        cand = dict(mA1=mA1, mA2=mA2, mB1=mB1, mB2=mB2, theta=theta, mis1=mis1, mis2=mis2, score=score)
                        if (best is None) or (score < best['score']): best = cand
                if best is None: raise RuntimeError("No diagonal match found.")
                self.sA = diagonal_supercell(sA0, best['mA1'], best['mA2'])
                Btmp = sB0.copy()
                if abs(best['theta']) > 1e-6: Btmp.rotate(best['theta'], 'z', rotate_cell=True, center='COM')
                self.sB = diagonal_supercell(Btmp, best['mB1'], best['mB2'])
                set_inplane_to_reference(self.sB, self.sA.cell.array.copy(), mode=self.strain_mode.currentText())
                self.iface = stack_A_on_B(self.sA, self.sB, gap=self.gap.value(), c_pad=self.c_pad.value())
                self.log(f"Diagonal: θ={best['theta']:.2f}°, mis=({best['mis1']:.2f}%, {best['mis2']:.2f}%), A=({best['mA1']},{best['mA2']}), B=({best['mB1']},{best['mB2']})")
            else:
                sA0, sB0 = trim_slab_by_thickness(slabA_copy, self.max_thickA.value(), 'top'), trim_slab_by_thickness(slabB_copy, self.max_thickB.value(), 'bottom')
                area_lim = self.area_limit.value() if self.area_limit.value() > 0 else None
                best = search_full_integer_supercells(sA0, sB0, max_entry=self.max_entry.value(), det_max=self.det_max.value(), angle_step=self.angle_step.value(), angle_tol=self.angle_tol.value(), max_mismatch=self.max_mismatch.value(), area_limit=area_lim, prefer_small=self.prefer_small.isChecked(), try_mirror=self.try_mirror.isChecked(), swap_roles=self.swap_roles.isChecked())
                ref, other = (sA0, sB0) if best['ref_is_A'] else (sB0, sA0)
                U_ref, V_other = (best['U'], best['V']) if best['ref_is_A'] else (best['V'], best['U'])
                s_ref = make_supercell(ref, [[U_ref[0,0], U_ref[0,1], 0], [U_ref[1,0], U_ref[1,1], 0], [0,0,1]])
                s_other = make_supercell(other, [[V_other[0,0], V_other[0,1], 0], [V_other[1,0], V_other[1,1], 0], [0,0,1]])
                self.sA, self.sB = (s_ref, s_other) if best['ref_is_A'] else (s_other, s_ref)
                if abs(best['theta']) > 1e-6: self.sB.rotate(best['theta'], 'z', rotate_cell=True, center='COM')
                set_inplane_to_reference(self.sB, self.sA.cell.array.copy(), mode=self.strain_mode.currentText())
                self.iface = stack_A_on_B(self.sA, self.sB, gap=self.gap.value(), c_pad=self.c_pad.value())
                self.log(f"Full: θ={best['theta']:.2f}°, mis=({best['mis1']:.2f}%, {best['mis2']:.2f}%), det(U)={best['mA']}, det(V)={best['mB']}, area≈{best['areaA']:.1f} Å²")
        except Exception as e:
            self.log(f"Matching error: {e}"); self.iface, self.sA, self.sB = Atoms(), Atoms(), Atoms()
            self.update_view(); return False
        self.update_view()
        k_iface = kmesh_from_cell(self.iface.cell, self.kspacing.value(), self.kz_min.value(), self.kz_fixed.value() or None)
        ntyp = len(set(self.iface.get_chemical_symbols())); self.info_ntyp_label.setText(f"Ntyp: {ntyp}")
        self.info.appendPlainText(f"Interface cell (Å): a={np.linalg.norm(self.iface.cell[0]):.3f}, b={np.linalg.norm(self.iface.cell[1]):.3f}, c={np.linalg.norm(self.iface.cell[2]):.3f}\n"
                                f"Suggested kpts ≈ {k_iface}\n"
                                f"Atom types (ntyp) = {ntyp}\n")
        return True
    def update_view(self):
        try:
            sel = self.model_combo.currentText(); atoms_to_show = None
            if sel == "Interface": atoms_to_show = self.iface
            elif sel.endswith("Slab A"): atoms_to_show = self.sA
            elif sel.endswith("Slab B"): atoms_to_show = self.sB
            if atoms_to_show is None: self.plotter.clear(); self.info_ntyp_label.setText("Ntyp: -"); return
            add_atoms_actor(self.plotter, atoms_to_show)
            self.info_ntyp_label.setText(f"Ntyp: {len(set(atoms_to_show.get_chemical_symbols()))}")
        except Exception as e:
            self.log(f"--- 3D VIEW ERROR ---\n"
                     f"Failed to render structure. Check PyVista/VTK installation and graphics drivers.\n"
                     f"Console may have more details.\nError: {e}\n--------------------")
            print(f"Detailed 3D View Error: {e}", file=sys.stderr)
    def export_qe(self):
        if self.iface is None or len(self.iface) == 0: QtWidgets.QMessageBox.information(self, "No model", "Please Build a valid structure first."); return
        outdir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output workdir", self.workdir.text())
        if not outdir: return
        out_path = Path(outdir)
        try:
            elements = collect_elements([self.iface]); pseudos = find_pseudos(Path(self.pseudo_dir.text().strip()), elements)
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Pseudo mapping error", str(e)); return
        base = f"A_h{self.hkl1.text().strip().replace(' ','')}_B_h{self.hkl2.text().strip().replace(' ','')}"
        d_iface, d_slabA, d_slabB = out_path/"interface", out_path/"slab_A", out_path/"slab_B"
        expA, expB = (self.sA, self.sB) if self.sA is not None else (self.slabA, self.slabB)
        spin_mags = default_spin_mags(elements) if self.enable_spin.isChecked() else None
        if self.enable_spin.isChecked():
            self.log("Spin polarization enabled. Applying presets for magnetic elements.")
            if not spin_mags: self.log("Warning: Spin polarization enabled, but no magnetic elements with presets were found.")
        
        common_params = { 'pseudo_dir': Path(self.pseudo_dir.text().strip()), 'pseudos': pseudos, 
                          'ecutwfc': self.ecutwfc.value(), 'ecutrho': self.ecutrho.value(),
                          'degauss': self.degauss.value(), 'smearing': self.smearing.currentText(), 
                          'spin_mags': spin_mags, 'mixing_mode': self.mixing_mode.currentText(),
                          'mixing_beta': self.mixing_beta.value(), 'conv_thr': self.conv_thr.text() }
        try:
            d_iface.mkdir(parents=True, exist_ok=True); d_slabA.mkdir(parents=True, exist_ok=True); d_slabB.mkdir(parents=True, exist_ok=True)
            k_iface = self.get_kpts(self.iface.cell)
            relax_params_iface = {**common_params, 'prefix': f'{base}_IFACE', 'calculation': 'vc-relax' if self.vcrelax.isChecked() else 'relax'}
            scf_params_iface = {**common_params, 'prefix': f'{base}_IFACE', 'calculation': 'scf'}
            write_qe_input_manual(self.iface, str(d_iface/f"{base}_IFACE_relax.in"), k_iface, relax_params_iface)
            write_qe_input_manual(self.iface, str(d_iface/f"{base}_IFACE_scf.in"), k_iface, scf_params_iface)
            k_A = self.get_kpts(expA.cell)
            relax_params_A = {**common_params, 'prefix': f'{base}_A', 'calculation': 'relax'}; scf_params_A = {**common_params, 'prefix': f'{base}_A', 'calculation': 'scf'}
            write_qe_input_manual(expA, str(d_slabA/f"{base}_A_relax.in"), k_A, relax_params_A)
            write_qe_input_manual(expA, str(d_slabA/f"{base}_A_scf.in"), k_A, scf_params_A)
            k_B = self.get_kpts(expB.cell)
            relax_params_B = {**common_params, 'prefix': f'{base}_B', 'calculation': 'relax'}; scf_params_B = {**common_params, 'prefix': f'{base}_B', 'calculation': 'scf'}
            write_qe_input_manual(expB, str(d_slabB/f"{base}_B_relax.in"), k_B, relax_params_B)
            write_qe_input_manual(expB, str(d_slabB/f"{base}_B_scf.in"), k_B, scf_params_B)
            write(str(d_iface/f"{base}_IFACE.cif"), self.iface); write(str(d_slabA/f"{base}_A.cif"), expA); write(str(d_slabB/f"{base}_B.cif"), expB)
            run_all = out_path/"run_all.sh"
            with open(run_all, "w", encoding="utf-8") as f:
                f.write("#!/usr/bin/env bash\nset -e\n\n"); f.write('PWX="${PWX:-pw.x}"\n')
                for sub, pref in [("interface", f"{base}_IFACE"), ("slab_A", f"{base}_A"), ("slab_B", f"{base}_B")]:
                    f.write(f'echo "==> {sub} relax"\n'); f.write(f'cd "{sub}" && $PWX < {pref}_relax.in > {pref}_relax.out && cd ..\n')
                    f.write(f'echo "==> {sub} scf"\n'); f.write(f'cd "{sub}" && $PWX < {pref}_scf.in > {pref}_scf.out && cd ..\n')
                f.write('echo "All done."\n')
            os.chmod(run_all, 0o755)
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Export error", str(e)); return
        QtWidgets.QMessageBox.information(self, "Done", f"Inputs and CIF files written under: {out_path}")
    
    def get_kpts(self, cell) -> Tuple[int, int, int]:
        if self.kpoint_mode_combo.currentText() == "Manual":
            try:
                parts = [int(p) for p in self.kpoint_manual_edit.text().split()]
                if len(parts) != 3: raise ValueError("Must have 3 integers")
                return tuple(parts)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "K-Point Error", f"Invalid manual k-point string. Must be 3 integers separated by spaces (e.g., '4 4 1').\nError: {e}")
                return (1, 1, 1) # Fallback
        else:
            kzfix = self.kz_fixed.value() or None
            return kmesh_from_cell(cell, self.kspacing.value(), self.kz_min.value(), kzfix)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(40,40,45)); pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(30,30,35)); pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45,45,50))
    pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white); pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white); pal.setColor(QtGui.QPalette.Button, QtGui.QColor(60,60,65))
    pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white); pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(76,163,224))
    pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(pal)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
