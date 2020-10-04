import uuid
import h5py
import subprocess
import numpy as np
import simtk.unit as u
from MDAnalysis.analysis import distances


def write_contact_map_h5(file_name, rows, cols):

    # Helper function to create ragged array
    def ragged(data):
        a = np.empty(len(data), dtype=object)
        a[...] = data
        return a

    # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype("int16"))

    with h5py.File(file_name, "w", swmr=False) as h5_file:
        # list of np arrays of shape (2 * X) where X varies
        data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
        h5_file.create_dataset(
            "contact_map",
            data=data,
            chunks=(1,) + data.shape[1:],
            dtype=dt,
            fletcher32=True,
        )


class SparseContactMapReporter:
    def __init__(
        self,
        file,
        reportInterval,
        selection="CA",
        threshold=8.0,
        batch_size=2,  # 1024,
        senders=[],
    ):

        self._file_idx = 0
        self._base_name = file
        self._report_interval = reportInterval
        self._selection = selection
        self._threshold = threshold
        self._batch_size = batch_size
        self._senders = senders

        self._init_batch()

    def _init_batch(self):
        # Frame counter for writing batches to HDF5
        self._num_frames = 0
        # Row, Column indices for contact matrix in COO format
        self._rows, self._cols = [], []

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _report_contact_maps(self, positions):

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

        self._report_contact_maps(positions)

        self._num_frames += 1

        if self._num_frames == self._batch_size:
            file_name = f"{self._base_name}_{self._file_idx:05d}_{uuid.uuid4()}.h5"
            write_contact_map_h5(file_name, self._rows, self._cols)
            self._init_batch()
            self._file_idx += 1

            for sender in self._senders:
                sender.send(file_name)
