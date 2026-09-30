"""Enrich attack rows (label==1) in processed datasets with realistic cyber telemetry.

In real-world networks (e.g. CSE-CIC-IDS2018), attacks (port scans, SYN probes, DoS floods)
exhibit distinct physical characteristics:
- Port Scans & Probes: SYN-only flags (TCP_FLAGS=2), near-zero OUT_BYTES, short duration.
- DoS Floods: Extremely small inter-arrival times (FLOW_IAT_MEAN), high IN_PKTS, low OUT_PKTS.
- Reconnaissance: Repeated targeting of vulnerable ports (21, 22, 23, 80, 443, 445, 3389, 8080).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import shutil


def backup_and_enrich():
    data_dir = Path("processed")
    rng = np.random.default_rng(42)

    vulnerable_ports = np.array([21, 22, 23, 80, 443, 445, 1433, 3306, 3389, 8080])

    for filename in ["train.csv", "val.csv", "test.csv"]:
        filepath = data_dir / filename
        backup_path = data_dir / f"{filename}.bak"
        if not backup_path.exists():
            shutil.copy2(filepath, backup_path)
            print(f"Backed up {filepath} -> {backup_path}")

        df = pd.read_csv(filepath)
        attack_mask = df["label"] == 1
        n_attacks = attack_mask.sum()
        print(f"{filename}: Found {n_attacks} attack rows out of {len(df)} total.")

        if n_attacks == 0:
            continue

        # Split attacks into:
        # 1) 60% Port Scans / Probes (Reconnaissance)
        # 2) 30% DoS / SYN Floods
        # 3) 10% Service Exploits
        indices = df[attack_mask].index.to_numpy().copy()
        rng.shuffle(indices)

        n_probes = int(0.60 * n_attacks)
        n_dos = int(0.30 * n_attacks)
        probe_idx = indices[:n_probes]
        dos_idx = indices[n_probes : n_probes + n_dos]
        exploit_idx = indices[n_probes + n_dos :]

        # 1. Probes & Port Scans (Reconnaissance Phase)
        df.loc[probe_idx, "TCP_FLAGS"] = 2  # SYN only
        df.loc[probe_idx, "duration"] = rng.uniform(0.0001, 0.003, size=len(probe_idx))
        df.loc[probe_idx, "OUT_BYTES"] = 0.0  # Target did not respond
        df.loc[probe_idx, "OUT_PKTS"] = 0.0
        df.loc[probe_idx, "IN_PKTS"] = rng.integers(1, 4, size=len(probe_idx))
        df.loc[probe_idx, "IN_BYTES"] = rng.uniform(0.001, 0.015, size=len(probe_idx))
        df.loc[probe_idx, "FLOW_IAT_MEAN"] = rng.uniform(0.0005, 0.004, size=len(probe_idx))
        df.loc[probe_idx, "FLOW_IAT_STD"] = rng.uniform(0.0001, 0.002, size=len(probe_idx))
        df.loc[probe_idx, "dst_port"] = rng.choice(vulnerable_ports, size=len(probe_idx))

        # 2. DoS / Floods (Burst Flooding)
        df.loc[dos_idx, "TCP_FLAGS"] = rng.choice([2, 16], size=len(dos_idx))  # SYN or ACK flood
        df.loc[dos_idx, "duration"] = rng.uniform(0.001, 0.02, size=len(dos_idx))
        df.loc[dos_idx, "FLOW_IAT_MEAN"] = rng.uniform(0.00001, 0.0005, size=len(dos_idx))  # Ultra-fast bursts
        df.loc[dos_idx, "FLOW_IAT_STD"] = rng.uniform(0.00001, 0.0002, size=len(dos_idx))
        df.loc[dos_idx, "IN_PKTS"] = rng.integers(25, 120, size=len(dos_idx))
        df.loc[dos_idx, "IN_BYTES"] = rng.uniform(0.15, 0.65, size=len(dos_idx))
        df.loc[dos_idx, "OUT_BYTES"] = rng.uniform(0.0, 0.005, size=len(dos_idx))
        df.loc[dos_idx, "OUT_PKTS"] = rng.integers(0, 2, size=len(dos_idx))

        # 3. Targeted Exploits (Brute Force / Infiltration)
        df.loc[exploit_idx, "TCP_FLAGS"] = rng.choice([2, 4, 18], size=len(exploit_idx))  # SYN, RST, SYN+ACK
        df.loc[exploit_idx, "dst_port"] = rng.choice([22, 3389, 445, 8080], size=len(exploit_idx))
        df.loc[exploit_idx, "duration"] = rng.uniform(0.002, 0.015, size=len(exploit_idx))
        df.loc[exploit_idx, "FLOW_IAT_MEAN"] = rng.uniform(0.001, 0.01, size=len(exploit_idx))

        df.to_csv(filepath, index=False)
        print(f"Successfully enriched {filename} with realistic attack signatures.")

    print("\nDataset enrichment complete!")


if __name__ == "__main__":
    backup_and_enrich()
