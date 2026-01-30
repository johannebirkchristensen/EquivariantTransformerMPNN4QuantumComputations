from time import time
from ase.db import connect
import os
import torch
from torch.utils.data import Dataset, DataLoader
from ase.db import connect
from ase import Atoms

class QM9Dataset(Dataset):
    def __init__(self, db_path, transform=None, max_samples=1000):
        """
        Args:
            db_path (str)
            transform (callable, optional)
            max_samples (int, optional): if set, limit to this many valid DB rows
        """
     
        self.db = connect(db_path)
  
        # Collect *actual* ASE DB row ids (safe when row ids are non-contiguous)
        n = self.db.count()
        valid_ids = list(range(1, n + 1))  # ASE DB ids start at 1

        #t0 = time.time()
       
        #valid_ids = [row.id for row in self.db.select()]  # row.id is the true DB id
        if max_samples is not None:
            #print(f"reduced dataset from {len(valid_ids)} ")
            valid_ids = valid_ids[:max_samples]
            #print(f"to {len(valid_ids)} ")
        #valid_ids = [row.id for row in self.db.select()]
        #print("ID scan time:", time.time() - t0)
        
        
        
        self.keys = valid_ids
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]           # actual ASE DB id
        try:
            row = self.db.get(id=key)  # use named arg id= to be explicit
        except KeyError:
            # print helpful debug info and re-raise so you still see the traceback
            print("DEBUG: invalid key:", key, "idx:", idx)
            raise

        atoms: Atoms = row.toatoms()

        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)

        #targets = torch.tensor([row.data['U']], dtype=torch.float32)
        targets = torch.tensor([
            row.data['mu'],      # 1. Dipole moment
            row.data['alpha'],   # 2. Polarizability
            row.data['homo'],    # 3. HOMO energy
            row.data['lumo'],    # 4. LUMO energy
            row.data['gap'],     # 5. HOMO-LUMO gap
            row.data['r2'],      # 6. Electronic spatial extent
            row.data['zpve'],    # 7. Zero point vibrational energy
            row.data['U0'],      # 8. Internal energy at 0K
            row.data['U'],       # 9. Internal energy at 298K
            row.data['H'],       # 10. Enthalpy at 298K
            row.data['G'],       # 11. Free energy at 298K
            row.data['Cv'],      # 12. Heat capacity at 298K
        ], dtype=torch.float32)  # Shape: [12]
        sample = {
            'atomic_numbers': atomic_numbers,
            'pos': pos,
            'natoms': len(atomic_numbers),
            'targets': targets,
            'batch': torch.zeros(len(atomic_numbers), dtype=torch.long)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def collate_mol(batch):
    atomic_numbers = torch.cat([b['atomic_numbers'] for b in batch], dim=0)
    pos = torch.cat([b['pos'] for b in batch], dim=0)
    natoms = torch.tensor([b['natoms'] for b in batch], dtype=torch.long)
    batch_tensor = torch.cat([b['batch'] + i for i, b in enumerate(batch)], dim=0)
    targets = torch.stack([b['targets'] for b in batch], dim=0)

    return {
        'atomic_numbers': atomic_numbers,
        'pos': pos,
        'natoms': natoms,
        'batch': batch_tensor,
        'targets': targets
    }



def get_qm9_loaders(db_path, batch_size=16, val_split=0.1, test_split=0.1,
                    shuffle=True, max_samples=100, num_workers=0):
    """
    Returns train, val, test loaders. Set max_samples for quick debugging.
    Use num_workers=0 when debugging to get clear tracebacks in the main process.
    """
    dataset = QM9Dataset(db_path, max_samples=max_samples)
    
    total = len(dataset)
    #print(f"Total samples in dataset: {total}")
    n_val = int(val_split * total)
    n_test = int(test_split * total)
    n_train = total - n_val - n_test

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    #print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}, Test samples: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                              collate_fn=collate_mol, num_workers=num_workers)
    #print("train_loader created with batch size:", batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_mol, num_workers=num_workers)
    #print("val_loader created with batch size:", batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_mol, num_workers=num_workers)
   # print("test_loader created with batch size:", batch_size)

    return train_loader, val_loader, test_loader
