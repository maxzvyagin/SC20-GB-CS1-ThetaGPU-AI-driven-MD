import uuid
import h5py
import logging
import numpy as np
import simtk.unit as u
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms

logger = logging.getLogger(__name__)


def write_contact_map_h5(h5_file, rows, cols):

    # Helper function to create ragged array
    def ragged(data):
        a = np.empty(len(data), dtype=object)
        a[...] = data
        return a

    # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype("int16"))

    # list of np arrays of shape (2 * X) where X varies
    data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
    h5_file.create_dataset(
        "contact_map",
        data=data,
        dtype=dt,
        fletcher32=True,
        chunks=(1,) + data.shape[1:],
    )


def write_rmsd(h5_file, rmsd):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )

def wrap(atoms):

    def wrap_nsp10_16(positions):
        # update the positions 
        atoms.positions = positions
        # only porting CA into nsp16
        nsp16 = atoms.segments[0].atoms
        # wrapping atoms into continous frame pbc box
        box_edge = nsp16.dimensions[0]
        box_center = box_edge / 2
        trans_vec = box_center - np.array(nsp16.center_of_mass())
        atoms.translate(trans_vec).wrap()
        trans_vec = box_center - np.array(atoms.center_of_mass())
        atoms.translate(trans_vec).wrap()

        return atoms.positions

    return wrap_nsp10_16


class SparseContactMapReporter:
    def __init__(
        self,
        file,
        reportInterval,
        wrap_pdb_file=None,
        reference_pdb_file=None,
        selection="CA",
        threshold=8.0,
        batch_size=2,  # 1024,
        senders=[],
    ):

        self._file_idx = 0
        self._base_name = file
        self._reference_pdb_file = reference_pdb_file
        self._report_interval = reportInterval
        self._selection = selection
        self._threshold = threshold
        self._batch_size = batch_size
        self._senders = senders

        self._init_batch()

        # Set up for reporting optional RMSD to reference state
        if reference_pdb_file is not None:
            u = mda.Universe(self._reference_pdb_file)
            # Convert openmm atom selection to mda
            selection = f"protein and name {self._selection}"
            self._reference_positions = u.select_atoms(selection).positions.copy()
            self._rmsd = []
        else:
            self._reference_positions = None

        if wrap_pdb_file is not None:
            u = mda.Universe(wrap_pdb_file)
            selection = f"protein and name {self._selection}"
            atoms = u.select_atoms(selection)
            self.wrap = wrap(atoms)
        else:
            self.wrap = None

    def _init_batch(self):
        # Frame counter for writing batches to HDF5
        self._num_frames = 0
        # Row, Column indices for contact matrix in COO format
        self._rows, self._cols = [], []

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _collect_rmsd(self, positions):
        if self.wrap is not None:
            positions = self.wrap(positions)

        rmsd = rms.rmsd(positions, self._reference_positions, superposition=True)
        self._rmsd.append(rmsd)

    def _collect_contact_maps(self, positions):

        contact_map = distances.contact_matrix(
            positions, self._threshold, returntype="sparse"
        )

        # Represent contact map in COO sparse format
        coo = contact_map.tocoo()
        self._rows.append(coo.row.astype("int16"))
        self._cols.append(coo.col.astype("int16"))

    def report(self, simulation, state):
        atom_indices = [
            a.index for a in simulation.topology.atoms() if a.name == self._selection
        ]
        all_positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        positions = all_positions[atom_indices].astype(np.float32)

        self._collect_contact_maps(positions)

        if self._reference_positions is not None:
            self._collect_rmsd(positions)

        self._num_frames += 1

        if self._num_frames == self._batch_size:
            file_name = f"{self._base_name}_{self._file_idx:05d}_{uuid.uuid4()}.h5"

            with h5py.File(file_name, "w", swmr=False) as h5_file:
                write_contact_map_h5(h5_file, self._rows, self._cols)
                # Optionally, write rmsd to the reference state
                if self._reference_positions is not None:
                    write_rmsd(h5_file, self._rmsd)
                    self._rmsd = []

            logger.debug(f"H5 file written with {len(self._rows)} frames.")
            self._init_batch()
            self._file_idx += 1

            for sender in self._senders:
                sender.send(file_name)
