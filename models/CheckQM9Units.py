"""
QM9 Unit Checker
================
Checks what units your QM9 database uses and compares with paper
"""

from ase.db import connect
import sys

if len(sys.argv) < 2:
    print("Usage: python check_qm9_units.py <path_to_qm9.db>")
    sys.exit(1)

db_path = sys.argv[1]
db = connect(db_path)

# Get first molecule
row = db.get(1)

print("="*70)
print("QM9 DATABASE UNIT CHECK")
print("="*70)

print("\nFirst molecule properties:")
print(f"  α (alpha):  {row.data['alpha']:.6f}")
print(f"  Δε (gap):   {row.data['gap']:.6f}")
print(f"  ε_HOMO:     {row.data['homo']:.6f}")
print(f"  ε_LUMO:     {row.data['lumo']:.6f}")
print(f"  μ (mu):     {row.data['mu']:.6f}")
print(f"  C_v:        {row.data['Cv']:.6f}")
print(f"  G:          {row.data['G']:.6f}")
print(f"  H:          {row.data['H']:.6f}")
print(f"  R²:         {row.data['r2']:.6f}")
print(f"  U:          {row.data['U']:.6f}")
print(f"  U₀:         {row.data['U0']:.6f}")
print(f"  ZPVE:       {row.data['zpve']:.6f}")

print("\n" + "="*70)
print("UNIT CONVERSIONS NEEDED")
print("="*70)

# QM9 database typically stores in these units:
# - Energies: Hartree (need to convert to meV)
# - Distances: Bohr (need to convert to Bohr or keep as is)
# - Dipole: e*Bohr (need to convert to Debye)
# - Polarizability: Bohr³ (paper uses Bohr³, so OK)
# - Heat capacity: cal/mol/K (already correct)

print("\nLikely units in YOUR database:")
print("  α:      Bohr³ (a₀³) - probably OK as-is")
print("  Δε:     Hartree → needs *27211.4 to get meV")
print("  ε_HOMO: Hartree → needs *27211.4 to get meV")
print("  ε_LUMO: Hartree → needs *27211.4 to get meV")
print("  μ:      e*Bohr → needs *2.5417464 to get Debye")
print("  C_v:    cal/mol/K - probably OK as-is")
print("  G:      Hartree → needs *27211.4 to get meV")
print("  H:      Hartree → needs *27211.4 to get meV")
print("  R²:     Bohr² (a₀²) - probably OK as-is")
print("  U:      Hartree → needs *27211.4 to get meV")
print("  U₀:     Hartree → needs *27211.4 to get meV")
print("  ZPVE:   Hartree → needs *27211.4 to get meV")

print("\nCONVERSION FACTORS:")
print("  1 Hartree = 27211.4 meV")
print("  1 e*Bohr = 2.5417464 Debye")

print("\n" + "="*70)
print("EXPECTED RANGES (from paper)")
print("="*70)

print("\nTypical QM9 values IN PAPER UNITS:")
print("  α:      ~70-80 Bohr³")
print("  Δε:     ~2000-8000 meV")
print("  ε_HOMO: ~-250 to -150 meV (NEGATIVE!)")
print("  ε_LUMO: ~-50 to +50 meV")
print("  μ:      ~0-5 Debye")
print("  C_v:    ~6-10 cal/mol/K")
print("  G:      Large (meV)")
print("  H:      Large (meV)")
print("  R²:     ~1200-1500 Bohr²")
print("  U:      Large (meV)")
print("  U₀:     Large (meV)")
print("  ZPVE:   ~80-130 meV")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

# Check if values need conversion
if abs(row.data['gap']) < 1:
    print("\n⚠️  Δε is < 1 → Likely in Hartree, needs conversion to meV")
else:
    print("\n✓ Δε is > 1 → Might already be in meV")

if abs(row.data['homo']) < 1:
    print("⚠️  ε_HOMO is < 1 → Likely in Hartree, needs conversion to meV")
else:
    print("✓ ε_HOMO is > 1 → Might already be in meV")

if row.data['mu'] < 0.1:
    print("⚠️  μ is very small → Likely in e*Bohr, needs conversion to Debye")
else:
    print("✓ μ seems reasonable")

if row.data['Cv'] < 1:
    print("⚠️  C_v is < 1 → Might need conversion")
else:
    print("✓ C_v seems reasonable")

print("\n" + "="*70)   
            """
            targets_list = torch.tensor([
                row.data['mu'], 
                row.data['alpha'],   # 1. α - Polarizability
                row.data['gap'],     # 2. Δε - HOMO-LUMO gap
                row.data['homo'],    # 3. ε_HOMO - HOMO energy
                row.data['lumo'],    # 4. ε_LUMO - LUMO energy
                                    # 5. μ - Dipole moment
                row.data['Cv'],      # 6. C_v - Heat capacity at 298K
                row.data['G'],       # 7. G - Free energy at 298K
                row.data['H'],       # 8. H - Enthalpy at 298K
                row.data['r2'],      # 9. R² - Electronic spatial extent
                row.data['U'],       # 10. U - Internal energy at 298K
                row.data['U0'],      # 11. U₀ - Internal energy at 0K
                row.data['zpve'],    # 12. ZPVE - Zero point vibrational energy
            ], dtype=torch.float32)
        except KeyError as e:
            print(f"ERROR: Missing property {e} in database row {key}")
            print(f"Available keys: {row.data.keys()}")
            raise
        # === build targets robustly (paper order) ===
        targets_list = [
            float(self._get_row_value(row, ['alpha', 'alpha0', 'alpha_A'])),           # α - Polarizability (a0^3)
            float(self._get_row_value(row, ['gap', 'gap_eV', 'delta_e'])),             # Δε - HOMO-LUMO gap (eV -> will convert)
            float(self._get_row_value(row, ['homo', 'HOMO', 'eHOMO'])),                # ε_HOMO (eV)
            float(self._get_row_value(row, ['lumo', 'LUMO', 'eLUMO'])),                # ε_LUMO (eV)
            float(self._get_row_value(row, ['mu', 'dipole'])),                         # μ - Dipole moment (Debye)
            float(self._get_row_value(row, ['Cv', 'cv', 'heat_capacity'])),            # C_v - Heat capacity (cal/mol K)
            float(self._get_row_value(row, ['G', 'free_energy', 'G_mol'])),            # G - Free energy (eV)
            float(self._get_row_value(row, ['H', 'enthalpy'])),                        # H - Enthalpy (eV)
            float(self._get_row_value(row, ['r2', 'R2', 'r_squared'])),                # R^2 (a0^2)
            float(self._get_row_value(row, ['U', 'u', 'internal_energy'])),            # U (eV)
            float(self._get_row_value(row, ['U0', 'u0', 'internal_energy_0'])),        # U0 (eV)
            float(self._get_row_value(row, ['zpve', 'ZPVE'])),                         # ZPVE (eV)
        ]
          """