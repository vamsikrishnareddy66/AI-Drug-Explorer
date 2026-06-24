"""
protein.py
----------
Handles protein target lookup, validation, and enriched metadata
fetching from the RCSB PDB REST API.

Additions over the intermediate version:
- Chain information
- Ligand information
- Active site residue lookup (via RCSB annotations)
- Reference ligand mapping for select PDB IDs
"""

import re
import requests

RCSB_ENTRY_API   = "https://data.rcsb.org/rest/v1/core/entry/{}"
RCSB_POLYMER_API = "https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}"
RCSB_NONPOLY_API = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{}/{}"
RCSB_FILE_DL     = "https://files.rcsb.org/download/{}.pdb"
REQUEST_TIMEOUT  = 10

# ── Reference ligands for known drug-target complexes ─────────────────────
REFERENCE_LIGANDS = {
    "1EQG": {
        "name": "Ibuprofen",
        "formula": "C13H18O2",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "mw": 206.28,
        "notes": "Co-crystallized NSAID inhibitor of COX-1 (Cyclooxygenase-1). "
                 "Ibuprofen occupies the COX active site channel; this structure "
                 "is a classic reference for NSAID binding studies.",
        "pubchem_cid": 3672,
    },
    "2HU4": {
        "name": "Oseltamivir (Tamiflu)",
        "formula": "C16H28N2O4",
        "smiles": "CCOC(=O)C1=CC(NC(C)=O)C(OC(CC)CC)CC1N",
        "mw": 312.40,
        "notes": "Co-crystallized neuraminidase inhibitor; FDA-approved influenza drug.",
        "pubchem_cid": 65028,
    },
    "6LU7": {
        "name": "N3 (Inhibitor)",
        "formula": "C34H50N6O9S",
        "smiles": "CC(C)[C@@H](NC(=O)[C@@H](CC(=O)N)NC(=O)c1cccc2ccccc12)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C=O)CCc1ccccc1",
        "mw": 722.86,
        "notes": "Peptide-based inhibitor of SARS-CoV-2 main protease.",
        "pubchem_cid": None,
    },
    "1HVR": {
        "name": "XK263 (Cyclic Urea Inhibitor)",
        "formula": "C28H35N4O5",
        "smiles": "O=C1NC(=O)C(Cc2ccccc2)(Cc2ccc(O)cc2)[C@@H]1C[C@@H]1O[C@@H]2CC(O)C[C@H]2O1",
        "mw": 539.62,
        "notes": "Cyclic urea-based HIV-1 protease inhibitor.",
        "pubchem_cid": None,
    },
}

# ── Known active site residues ─────────────────────────────────────────────
# Residues within 5 Å of the co-crystallized ligand (or curated annotations)
ACTIVE_SITE_RESIDUES = {
    "1EQG": [
        "TYR355", "ARG120", "TYR385", "TRP387", "PHE381",
        "LEU384", "PHE205", "PHE209", "SER530", "VAL349",
    ],  # COX-1 residues within 5 Å of Ibuprofen (IBP) from 1EQG
    "2HU4": ["ARG118", "ARG152", "ARG224", "GLU119", "ARG292", "ARG371", "TYR406"],
    "6LU7": ["HIS41", "CYS145", "MET49", "MET165", "GLU166", "HIS163", "GLY143"],
    "1HVR": ["ASP25", "ASP25'", "THR26", "GLY27", "ALA28", "ILE50", "GLY49"],
    "3POZ": ["LYS802", "ASP810", "ASN822", "GLU840", "ASP933"],
    "1ATP": ["LYS72", "GLU91", "ASP184", "ASN171", "GLY52", "GLY55"],
}

# ── Offline fallback cache ─────────────────────────────────────────────────
FALLBACK_PROTEINS = {
    "1EQG": {
        "name": "Cyclooxygenase-1 (COX-1) complexed with Ibuprofen",
        "organism": "Ovis aries (Sheep)",
        "method": "X-RAY DIFFRACTION",
        "resolution": 3.10,
        "chains": ["A", "B"],
        "ligands": ["Ibuprofen (co-crystallized ligand)"],
        "description": "Ovine COX-1 bound to the NSAID Ibuprofen. "
                       "Classic structure for studying NSAID binding at the COX active site.",
    },
    "6LU7": {
        "name": "SARS-CoV-2 Main Protease (Mpro) with N3 Inhibitor",
        "organism": "Severe acute respiratory syndrome coronavirus 2",
        "method": "X-RAY DIFFRACTION",
        "resolution": 2.16,
        "chains": ["A", "B"],
        "ligands": ["N3"],
        "description": "The SARS-CoV-2 main protease is essential for viral polyprotein processing.",
    },
    "2HU4": {
        "name": "Influenza Neuraminidase H5N1 with Oseltamivir",
        "organism": "Influenza A virus (A/Vietnam/1203/2004(H5N1))",
        "method": "X-RAY DIFFRACTION",
        "resolution": 2.50,
        "chains": ["A", "B", "C", "D"],
        "ligands": ["G39 (Oseltamivir carboxylate)"],
        "description": "H5N1 neuraminidase bound to oseltamivir carboxylate — key influenza drug target.",
    },
    "1HVR": {
        "name": "HIV-1 Protease with XK263 Inhibitor",
        "organism": "Human immunodeficiency virus 1",
        "method": "X-RAY DIFFRACTION",
        "resolution": 1.80,
        "chains": ["A", "B"],
        "ligands": ["XK2 (Cyclic Urea)"],
        "description": "HIV-1 protease is the classic antiviral drug target for AIDS treatment.",
    },
    "3POZ": {
        "name": "PI3K Alpha Kinase Domain",
        "organism": "Homo sapiens",
        "method": "X-RAY DIFFRACTION",
        "resolution": 2.40,
        "chains": ["A"],
        "ligands": ["PIK3CA inhibitor"],
        "description": "Phosphoinositide 3-kinase — major oncology drug target.",
    },
    "1ATP": {
        "name": "cAMP-Dependent Protein Kinase Catalytic Subunit",
        "organism": "Homo sapiens",
        "method": "X-RAY DIFFRACTION",
        "resolution": 1.80,
        "chains": ["A", "B", "I"],
        "ligands": ["ATP", "MN"],
        "description": "PKA catalytic subunit: the prototypic serine/threonine kinase.",
    },
}


def is_valid_pdb_id(pdb_id: str) -> bool:
    if not pdb_id:
        return False
    pdb_id = pdb_id.strip().upper()
    return bool(re.match(r"^[1-9][A-Z0-9]{3}$", pdb_id))


def get_protein_info(pdb_id: str) -> dict:
    """
    Fetch enriched protein metadata from the RCSB PDB REST API.
    Includes chains, ligands, and active site residues.
    Falls back to offline cache on network failure.
    """
    pdb_id = pdb_id.strip().upper()
    info = {
        "pdb_id": pdb_id,
        "structure_url": f"https://www.rcsb.org/structure/{pdb_id}",
        "source": "live",
        "chains": [],
        "ligands": [],
        "active_site_residues": ACTIVE_SITE_RESIDUES.get(pdb_id, []),
        "reference_ligand": REFERENCE_LIGANDS.get(pdb_id),
    }

    try:
        r = requests.get(RCSB_ENTRY_API.format(pdb_id), timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()

            # Title
            title = data.get("struct", {}).get("title", "Unknown title")

            # Organism — try entry-level field first, then polymer entity API
            organism = "Not specified"
            try:
                # Some entry responses include organism at root level
                src = data.get("rcsb_entity_source_organism")
                if src and isinstance(src, list):
                    sci = src[0].get("ncbi_scientific_name") or src[0].get("scientific_name")
                    if sci:
                        organism = sci
            except Exception:
                pass

            if organism == "Not specified":
                # Fetch from first polymer entity (entity 1)
                try:
                    pe_url = RCSB_POLYMER_API.format(pdb_id, 1)
                    pe_r = requests.get(pe_url, timeout=REQUEST_TIMEOUT)
                    if pe_r.status_code == 200:
                        pe_data = pe_r.json()
                        src_list = pe_data.get("rcsb_entity_source_organism", [])
                        if src_list:
                            sci = (src_list[0].get("ncbi_scientific_name")
                                   or src_list[0].get("scientific_name", ""))
                            if sci:
                                taxon = src_list[0].get("ncbi_common_names", [])
                                common = f" ({taxon[0]})" if taxon else ""
                                organism = sci + common
                except Exception:
                    pass

            # Use offline fallback organism if API still returns nothing
            if organism == "Not specified" and pdb_id in FALLBACK_PROTEINS:
                organism = FALLBACK_PROTEINS[pdb_id].get("organism", "Not specified")

            # Experimental method
            method = "Not specified"
            try:
                method = data.get("exptl", [{}])[0].get("method", "Not specified")
            except (KeyError, IndexError, TypeError):
                pass

            # Resolution
            resolution = None
            try:
                resolution = data["rcsb_entry_info"]["resolution_combined"][0]
            except (KeyError, IndexError, TypeError):
                pass

            # Chain / entity info
            num_entities = data.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 1)
            chain_ids = []
            try:
                assemblies = data.get("rcsb_assemblies", [])
                for asm in assemblies:
                    for chain in asm.get("rcsb_assembly_info", {}).get("polymer_entity_instance_ids", []):
                        if chain not in chain_ids:
                            chain_ids.append(chain)
            except Exception:
                pass

            # Ligand / non-polymer info
            ligand_names = []
            try:
                npe_count = data.get("rcsb_entry_info", {}).get("nonpolymer_entity_count", 0)
                if npe_count and npe_count > 0:
                    for i in range(1, min(int(npe_count) + 1, 8)):
                        try:
                            lr = requests.get(
                                RCSB_NONPOLY_API.format(pdb_id, i),
                                timeout=REQUEST_TIMEOUT
                            )
                            if lr.status_code == 200:
                                ld = lr.json()
                                chem = ld.get("chem_comp", {})
                                name = (chem.get("name") or chem.get("id") or "").strip()
                                comp_id = chem.get("id", "").upper()
                                # Exclude water and common solvents
                                if name and comp_id not in ("HOH", "H2O", "WAT", "SO4",
                                                             "PO4", "EDO", "GOL", ""):
                                    # Flag as co-crystallized if it matches reference ligand
                                    ref = REFERENCE_LIGANDS.get(pdb_id, {})
                                    ref_name = ref.get("name", "")
                                    # Match by name or PDB comp_id
                                    if (ref_name and ref_name.lower() in name.lower()) or \
                                       (comp_id in ("IBP", "IBU", "G39", "N3", "XK2", "ATP")):
                                        ligand_names.append(f"{name} (co-crystallized ligand)")
                                    else:
                                        ligand_names.append(name)
                        except Exception:
                            break
            except Exception:
                pass

            # If API returned no ligands but we have a known reference ligand, use it
            if not ligand_names and pdb_id in REFERENCE_LIGANDS:
                ref = REFERENCE_LIGANDS[pdb_id]
                ligand_names = [f"{ref['name']} (co-crystallized ligand)"]
            elif not ligand_names and pdb_id in FALLBACK_PROTEINS:
                ligand_names = FALLBACK_PROTEINS[pdb_id].get("ligands", [])

            info.update({
                "name": title,
                "organism": organism,
                "method": method,
                "resolution": resolution,
                "num_entities": num_entities or 1,
                "chains": chain_ids or [],
                "ligands": ligand_names,
                "description": f"Solved by {method}." + (
                    f" Resolution: {resolution} Å." if resolution else ""
                ),
            })
            return info

    except requests.exceptions.RequestException:
        pass

    # Offline fallback
    info["source"] = "offline_fallback"
    info["resolution"] = None
    info["num_entities"] = 1
    fb = FALLBACK_PROTEINS.get(pdb_id, {})
    if fb:
        info.update(fb)
        # Ensure reference ligand label is applied in fallback too
        if not info.get("ligands") and pdb_id in REFERENCE_LIGANDS:
            ref = REFERENCE_LIGANDS[pdb_id]
            info["ligands"] = [f"{ref['name']} (co-crystallized ligand)"]
    else:
        info.update({
            "name": "Details unavailable (offline)",
            "organism": "Not available",
            "method": "Not available",
            "description": "Could not reach RCSB PDB API and no local fallback exists.",
        })
        # Still apply reference ligand if we have it
        if pdb_id in REFERENCE_LIGANDS:
            ref = REFERENCE_LIGANDS[pdb_id]
            info["ligands"] = [f"{ref['name']} (co-crystallized ligand)"]
    return info


def download_structure_file(pdb_id: str):
    pdb_id = pdb_id.strip().upper()
    try:
        r = requests.get(RCSB_FILE_DL.format(pdb_id), timeout=15)
        if r.status_code == 200:
            return r.content, None
        return None, f"Structure file not found (HTTP {r.status_code})."
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"
