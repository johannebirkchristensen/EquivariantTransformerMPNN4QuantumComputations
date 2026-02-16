from torch_geometric.datasets import QM9

dataset = QM9(root='./qm9_data')

print("Atomref values for U₀, U, H, G:")
for target_idx, name in [(7, 'U₀'), (8, 'U'), (9, 'H'), (10, 'G')]:
    atomref = dataset.atomref(target=target_idx)
    print(f"\n{name} (target {target_idx}):")
    print(f"  H (Z=1):  {atomref[1, 0].item():.6f} eV")
    print(f"  C (Z=6):  {atomref[6, 0].item():.6f} eV")
    print(f"  N (Z=7):  {atomref[7, 0].item():.6f} eV")
    print(f"  O (Z=8):  {atomref[8, 0].item():.6f} eV")
    print(f"  F (Z=9):  {atomref[9, 0].item():.6f} eV")

# Check one molecule
data = dataset[0]
print(f"\nFirst molecule:")
print(f"  Atomic numbers: {data.z.tolist()}")
print(f"  Raw U₀: {data.y[0, 7].item():.6f} eV")

# Calculate atomref correction for U₀
atomref_u0 = dataset.atomref(target=7)
correction = sum(atomref_u0[z, 0].item() for z in data.z)
print(f"  Atomref correction (sum): {correction:.6f} eV")
print(f"  Corrected U₀: {data.y[0, 7].item() - correction:.6f} eV")