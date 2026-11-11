"""The methods guide: what each method, basis set, solvent model and conformer search is, as data.

One source for the GUI (the info text under each pipeline choice, Help > Methods guide...), the CLI
(``quantabench-cli methods [NAME]``) and ``docs/METHODS.md`` (written by ``scripts/write_methods_doc.py``; a
test fails when the committed file differs from this data).

Every engine in ``engines.all_engines()``, CREST level, optimizer, solvation model and the functionals and basis
sets of the QM page must have an entry (``tests/test_method_guide.py``), so a new engine without an explanation
fails a test. Add the entry here, map its key in ``ENGINE_ENTRIES`` and give it a line in the method's
``implementations``, then run ``python scripts/write_methods_doc.py``.

Text fields use a small markup, rendered for each output:

    `code`          a name to type or a column (code font; plain in the CLI)
    [[key]]         a link to another entry, shown with that entry's name
    [[key|text]]    a link shown as ``text``

References were checked against Crossref (authors, journal, year, volume, pages, DOI) on 2026-11-07. Numbers
marked "QuantaBench measured" come from docs/VALIDATION.md, docs/ENGINES.md and docs/dev/BACKLOG.md.
"""

from __future__ import annotations

import difflib
import html as _html
import re
import textwrap
import unicodedata
import urllib.parse
from dataclasses import dataclass, field, replace

FAMILIES: tuple[str, ...] = (
    "Machine-learning potentials",
    "Tight binding (xTB)",
    "Semi-empirical (NDDO)",
    "Force fields",
    "Conformer search",
    "DFT functionals and HF",
    "Basis sets",
    "Implicit solvation",
    "Optimizers",
    "Spartan's levels",
    "Property estimates",
    "Concepts",
)


@dataclass(frozen=True)
class Reference:
    authors: str        # surnames as the journal lists them ("et al." for long lists)
    title: str
    journal: str
    year: int
    volume: str
    pages: str
    doi: str

    @property
    def url(self) -> str:
        return "https://doi.org/" + urllib.parse.quote(self.doi, safe="/")

    def citation(self) -> str:
        where = ", ".join(x for x in (self.volume, self.pages) if x)
        return f"{self.authors}. {self.title}. {self.journal}{' ' + where if where else ''} ({self.year})."


@dataclass(frozen=True)
class Entry:
    key: str
    name: str
    family: str
    summary: str                      # one line: the GUI info text and the combo tooltips
    what: str                         # what it is, 2-4 short sentences
    good_for: str
    watch_out: str
    cost: str
    dispersion: str
    solvation: str
    references: tuple[Reference, ...] = ()
    aliases: tuple[str, ...] = ()     # other spellings that find this entry (case, hyphens and ω/w are loose anyway)
    see_also: tuple[str, ...] = ()
    implementations: tuple[tuple[str, str], ...] = ()   # (engine key, how that engine runs the method)
    # one line each, where it applies (``DECISIONS``): the error it corrects, what to use it for, what not to
    corrects: str = ""
    use_for: str = ""
    avoid_for: str = ""

    @property
    def anchor(self) -> str:
        """URL- and heading-safe id: '*' -> 's', '+' -> 'p', '/' -> '-' (Psi4's basis file names do the same)."""
        return re.sub(r"[^a-z0-9-]", "", self.key.lower().replace("*", "s").replace("+", "p").replace("/", "-"))

    def implementation(self, engine_key: str) -> str:
        return dict(self.implementations).get(engine_key, "")


# ============================================================================== references (checked on Crossref)
def _ref(authors, title, journal, year, volume, pages, doi) -> Reference:
    return Reference(authors, title, journal, year, volume, pages, doi)


R_AIMNET2 = _ref("Anstine, Zubatyuk, Isayev", "AIMNet2: a neural network potential to meet your neutral, charged, "
                 "organic, and elemental-organic needs", "Chem. Sci.", 2025, "16", "10228-10244", "10.1039/D4SC08572H")
R_GFN2 = _ref("Bannwarth, Ehlert, Grimme", "GFN2-xTB, an accurate and broadly parametrized self-consistent "
              "tight-binding quantum chemical method with multipole electrostatics and density-dependent dispersion "
              "contributions", "J. Chem. Theory Comput.", 2019, "15", "1652-1671", "10.1021/acs.jctc.8b01176")
R_GFN1 = _ref("Grimme, Bannwarth, Shushkov", "A robust and accurate tight-binding quantum chemical method for "
              "structures, vibrational frequencies, and noncovalent interactions of large molecular systems "
              "parametrized for all spd-block elements (Z = 1-86)", "J. Chem. Theory Comput.", 2017, "13",
              "1989-2009", "10.1021/acs.jctc.7b00118")
R_XTB_REVIEW = _ref("Bannwarth, Caldeweyher, Ehlert, Hansen, Pracht, Seibert, Spicher, Grimme",
                    "Extended tight-binding quantum chemistry methods", "WIREs Comput. Mol. Sci.", 2021, "11",
                    "e1493", "10.1002/wcms.1493")
R_DXTB = _ref("Friede, Hölzer, Ehlert, Grimme", "dxtb, an efficient and fully differentiable framework for "
              "extended tight-binding", "J. Chem. Phys.", 2024, "161", "062501", "10.1063/5.0216715")
R_GFNFF = _ref("Spicher, Grimme", "Robust atomistic modeling of materials, organometallic, and biochemical systems",
               "Angew. Chem. Int. Ed.", 2020, "59", "15665-15673", "10.1002/anie.202004239")
R_PM7 = _ref("Stewart", "Optimization of parameters for semiempirical methods VI: more modifications to the NDDO "
             "approximations and re-optimization of parameters", "J. Mol. Model.", 2013, "19", "1-32",
             "10.1007/s00894-012-1667-x")
R_PM6 = _ref("Stewart", "Optimization of parameters for semiempirical methods V: modification of NDDO "
             "approximations and application to 70 elements", "J. Mol. Model.", 2007, "13", "1173-1213",
             "10.1007/s00894-007-0233-4")
R_D3H4 = _ref("Řezáč, Hobza", "Advanced corrections of hydrogen bonding and dispersion for semiempirical quantum "
              "mechanical methods", "J. Chem. Theory Comput.", 2012, "8", "141-151", "10.1021/ct200751e")
R_D3H4X = _ref("Brahmkshatriya, Dobeš, Fanfrlík, Řezáč, Paruch, Bronowska, Lepšík, Hobza", "Quantum mechanical "
               "scoring: structural and energetic insights into cyclin-dependent kinase 2 inhibition by "
               "pyrazolo[1,5-a]pyrimidines", "Curr. Comput.-Aided Drug Des.", 2013, "9", "118-129",
               "10.2174/1573409911309010011")
R_MMFF94 = _ref("Halgren", "Merck molecular force field. I. Basis, form, scope, parameterization, and performance "
                "of MMFF94", "J. Comput. Chem.", 1996, "17", "490-519",
                "10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P")
R_MMFF94S = _ref("Halgren", "MMFF VI. MMFF94s option for energy minimization studies", "J. Comput. Chem.", 1999,
                 "20", "720-729", "10.1002/(SICI)1096-987X(199905)20:7<720::AID-JCC7>3.0.CO;2-X")
R_UFF = _ref("Rappé, Casewit, Colwell, Goddard, Skiff", "UFF, a full periodic table force field for molecular "
             "mechanics and molecular dynamics simulations", "J. Am. Chem. Soc.", 1992, "114", "10024-10035",
             "10.1021/ja00051a040")
R_ETKDG = _ref("Riniker, Landrum", "Better informed distance geometry: using what we know to improve conformation "
               "generation", "J. Chem. Inf. Model.", 2015, "55", "2562-2574", "10.1021/acs.jcim.5b00654")
R_ETKDG3 = _ref("Wang, Witek, Landrum, Riniker", "Improving conformer generation for small rings and macrocycles "
                "based on distance geometry and experimental torsional-angle preferences", "J. Chem. Inf. Model.",
                2020, "60", "2044-2058", "10.1021/acs.jcim.0c00025")
R_CREST = _ref("Pracht, Bohle, Grimme", "Automated exploration of the low-energy chemical space with fast quantum "
               "chemical methods", "Phys. Chem. Chem. Phys.", 2020, "22", "7169-7192", "10.1039/C9CP06869D")
R_CREST3 = _ref("Pracht et al.", "CREST, a program for the exploration of low-energy molecular chemical space",
                "J. Chem. Phys.", 2024, "160", "114110", "10.1063/5.0197592")
R_B3 = _ref("Becke", "Density-functional thermochemistry. III. The role of exact exchange", "J. Chem. Phys.", 1993,
            "98", "5648-5652", "10.1063/1.464913")
R_B3LYP = _ref("Stephens, Devlin, Chabalowski, Frisch", "Ab initio calculation of vibrational absorption and "
               "circular dichroism spectra using density functional force fields", "J. Phys. Chem.", 1994, "98",
               "11623-11627", "10.1021/j100096a001")
R_D3 = _ref("Grimme, Antony, Ehrlich, Krieg", "A consistent and accurate ab initio parametrization of density "
            "functional dispersion correction (DFT-D) for the 94 elements H-Pu", "J. Chem. Phys.", 2010, "132",
            "154104", "10.1063/1.3382344")
R_D3BJ = _ref("Grimme, Ehrlich, Goerigk", "Effect of the damping function in dispersion corrected density "
              "functional theory", "J. Comput. Chem.", 2011, "32", "1456-1465", "10.1002/jcc.21759")
R_D4 = _ref("Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher, Bannwarth, Grimme", "A generally applicable "
            "atomic-charge dependent London dispersion correction", "J. Chem. Phys.", 2019, "150", "154122",
            "10.1063/1.5090222")
R_DISP_REVIEW = _ref("Grimme, Hansen, Brandenburg, Bannwarth", "Dispersion-corrected mean-field electronic "
                     "structure methods", "Chem. Rev.", 2016, "116", "5105-5154", "10.1021/acs.chemrev.5b00533")
R_GMTKN55 = _ref("Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme", "A look at the density functional theory zoo "
                 "with the advanced GMTKN55 database for general main group thermochemistry, kinetics and "
                 "noncovalent interactions", "Phys. Chem. Chem. Phys.", 2017, "19", "32184-32215",
                 "10.1039/C7CP04913G")
R_ALKANE = _ref("Lüttschwager, Wassermann, Mata, Suhm", "The last globally stable extended alkane",
                "Angew. Chem. Int. Ed.", 2013, "52", "463-466", "10.1002/anie.201202894")
R_WB97XD = _ref("Chai, Head-Gordon", "Long-range corrected hybrid density functionals with damped atom-atom "
                "dispersion corrections", "Phys. Chem. Chem. Phys.", 2008, "10", "6615", "10.1039/B810189B")
R_WB97MV = _ref("Mardirossian, Head-Gordon", "ωB97M-V: a combinatorially optimized, range-separated hybrid, "
                "meta-GGA density functional with VV10 nonlocal correlation", "J. Chem. Phys.", 2016, "144",
                "214110", "10.1063/1.4952647")
R_PBE = _ref("Perdew, Burke, Ernzerhof", "Generalized gradient approximation made simple", "Phys. Rev. Lett.", 1996,
             "77", "3865-3868", "10.1103/PhysRevLett.77.3865")
R_PBE0 = _ref("Adamo, Barone", "Toward reliable density functional methods without adjustable parameters: the PBE0 "
              "model", "J. Chem. Phys.", 1999, "110", "6158-6170", "10.1063/1.478522")
R_R2SCAN = _ref("Furness, Kaplan, Ning, Perdew, Sun", "Accurate and numerically efficient r2SCAN "
                "meta-generalized gradient approximation", "J. Phys. Chem. Lett.", 2020, "11", "8208-8215",
                "10.1021/acs.jpclett.0c02405")
R_R2SCAN_D4 = _ref("Ehlert, Huniar, Ning, Furness, Sun, Kaplan, Perdew, Brandenburg", "r2SCAN-D4: dispersion "
                   "corrected meta-generalized gradient approximation for general chemical applications",
                   "J. Chem. Phys.", 2021, "154", "061101", "10.1063/5.0041008")
R_R2SCAN3C = _ref("Grimme, Hansen, Ehlert, Mewes", "r2SCAN-3c: a \"Swiss army knife\" composite "
                  "electronic-structure method", "J. Chem. Phys.", 2021, "154", "064103", "10.1063/5.0040021")
R_HF3C = _ref("Sure, Grimme", "Corrected small basis set Hartree-Fock method for large systems",
              "J. Comput. Chem.", 2013, "34", "1672-1685", "10.1002/jcc.23317")
R_GCP = _ref("Kruse, Grimme", "A geometrical correction for the inter- and intra-molecular basis set superposition "
             "error in Hartree-Fock and density functional theory calculations for large systems", "J. Chem. Phys.",
             2012, "136", "154101", "10.1063/1.3700154")
R_COUNTERPOISE = _ref("Boys, Bernardi", "The calculation of small molecular interactions by the differences of "
                      "separate total energies. Some procedures with reduced errors", "Mol. Phys.", 1970, "19",
                      "553-566", "10.1080/00268977000101561")
R_GXTB = _ref("Froitzheim, Müller, Hansen, Grimme", "g-xTB: a general-purpose extended tight-binding electronic "
              "structure method for the elements H to Lr (Z = 1-103)", "ChemRxiv (preprint)", 2025, "", "",
              "10.26434/chemrxiv-2025-bjxvt")
R_OECD_PFAS = _ref("OECD", "Reconciling terminology of the universe of per- and polyfluoroalkyl substances: "
                   "recommendations and practical guidance", "OECD Series on Risk Management of Chemicals", 2021,
                   "61", "", "10.1787/e458e796-en")
R_M062X = _ref("Zhao, Truhlar", "The M06 suite of density functionals for main group thermochemistry, "
               "thermochemical kinetics, noncovalent interactions, excited states, and transition elements: two new "
               "functionals and systematic testing of four M06-class functionals and 12 other functionals",
               "Theor. Chem. Acc.", 2008, "120", "215-241", "10.1007/s00214-007-0310-x")
R_GRID = _ref("Wheeler, Houk", "Integration grid errors for meta-GGA-predicted reaction energies: origin of grid "
              "errors for the M06 suite of functionals", "J. Chem. Theory Comput.", 2010, "6", "395-404",
              "10.1021/ct900639j")
R_ROOTHAAN = _ref("Roothaan", "New developments in molecular orbital theory", "Rev. Mod. Phys.", 1951, "23",
                  "69-89", "10.1103/RevModPhys.23.69")
R_KOOPMANS = _ref("Koopmans", "Über die Zuordnung von Wellenfunktionen und Eigenwerten zu den einzelnen Elektronen "
                  "eines Atoms", "Physica", 1934, "1", "104-113", "10.1016/S0031-8914(34)90011-2")
R_PSI4 = _ref("Smith et al.", "Psi4 1.4: open-source software for high-throughput quantum chemistry",
              "J. Chem. Phys.", 2020, "152", "184108", "10.1063/5.0006002")
R_DEF2 = _ref("Weigend, Ahlrichs", "Balanced basis sets of split valence, triple zeta valence and quadruple zeta "
              "valence quality for H to Rn: design and assessment of accuracy", "Phys. Chem. Chem. Phys.", 2005, "7",
              "3297", "10.1039/B508541A")
R_DEF2D = _ref("Rappoport, Furche", "Property-optimized Gaussian basis sets for molecular response calculations",
               "J. Chem. Phys.", 2010, "133", "134105", "10.1063/1.3484283")
R_631G = _ref("Hehre, Ditchfield, Pople", "Self-consistent molecular orbital methods. XII. Further extensions of "
              "Gaussian-type basis sets for use in molecular orbital studies of organic molecules", "J. Chem. Phys.",
              1972, "56", "2257-2261", "10.1063/1.1677527")
R_POLARIZATION = _ref("Hariharan, Pople", "The influence of polarization functions on molecular orbital "
                      "hydrogenation energies", "Theor. Chim. Acta", 1973, "28", "213-222", "10.1007/BF00533485")
R_DIFFUSE = _ref("Clark, Chandrasekhar, Spitznagel, Schleyer", "Efficient diffuse function-augmented basis sets for "
                 "anion calculations. III. The 3-21+G basis set for first-row elements, Li-F", "J. Comput. Chem.",
                 1983, "4", "294-301", "10.1002/jcc.540040303")
R_6311G = _ref("Krishnan, Binkley, Seeger, Pople", "Self-consistent molecular orbital methods. XX. A basis set for "
               "correlated wave functions", "J. Chem. Phys.", 1980, "72", "650-654", "10.1063/1.438955")
R_CCPVDZ = _ref("Dunning", "Gaussian basis sets for use in correlated molecular calculations. I. The atoms boron "
                "through neon and hydrogen", "J. Chem. Phys.", 1989, "90", "1007-1023", "10.1063/1.456153")
R_AUG = _ref("Kendall, Dunning, Harrison", "Electron affinities of the first-row atoms revisited. Systematic basis "
             "sets and wave functions", "J. Chem. Phys.", 1992, "96", "6796-6806", "10.1063/1.462569")
R_ALPB = _ref("Ehlert, Stahn, Spicher, Grimme", "Robust and efficient implicit solvation model for fast "
              "semiempirical methods", "J. Chem. Theory Comput.", 2021, "17", "4250-4261", "10.1021/acs.jctc.1c00471")
R_GB = _ref("Still, Tempczyk, Hawley, Hendrickson", "Semianalytical treatment of solvation for molecular mechanics "
            "and dynamics", "J. Am. Chem. Soc.", 1990, "112", "6127-6129", "10.1021/ja00172a038")
R_COSMO = _ref("Klamt, Schüürmann", "COSMO: a new approach to dielectric screening in solvents with explicit "
               "expressions for the screening energy and its gradient", "J. Chem. Soc., Perkin Trans. 2", 1993, "",
               "799-805", "10.1039/P29930000799")
R_PCM_REVIEW = _ref("Tomasi, Mennucci, Cammi", "Quantum mechanical continuum solvation models", "Chem. Rev.", 2005,
                    "105", "2999-3094", "10.1021/cr9904009")
R_IMPLICIT_REVIEW = _ref("Cramer, Truhlar", "Implicit solvation models: equilibria, structure, spectra, and "
                         "dynamics", "Chem. Rev.", 1999, "99", "2161-2200", "10.1021/cr960149m")
R_CPCM = _ref("Barone, Cossi", "Quantum calculation of molecular energies and energy gradients in solution by a "
              "conductor solvent model", "J. Phys. Chem. A", 1998, "102", "1995-2001", "10.1021/jp9716997")
R_DDX = _ref("Nottoli, Herbst, Mikhalev, Jha, Lipparini, Stamm", "ddX: polarizable continuum solvation from small "
             "molecules to proteins", "WIREs Comput. Mol. Sci.", 2024, "14", "e1726", "10.1002/wcms.1726")
R_SMD = _ref("Marenich, Cramer, Truhlar", "Universal solvation model based on solute electron density and on a "
             "continuum model of the solvent defined by the bulk dielectric constant and atomic surface tensions",
             "J. Phys. Chem. B", 2009, "113", "6378-6396", "10.1021/jp810292n")
R_LBFGS = _ref("Liu, Nocedal", "On the limited memory BFGS method for large scale optimization", "Math. Program.",
               1989, "45", "503-528", "10.1007/BF01589116")
R_FIRE = _ref("Bitzek, Koskinen, Gähler, Moseler, Gumbsch", "Structural relaxation made simple", "Phys. Rev. Lett.",
              2006, "97", "170201", "10.1103/PhysRevLett.97.170201")
R_GIROLAMI = _ref("Girolami", "A simple \"back of the envelope\" method for estimating the densities and molecular "
                 "volumes of liquids and solids", "J. Chem. Educ.", 1994, "71", "962", "10.1021/ed071p962")
R_BONDI = _ref("Bondi", "van der Waals volumes and radii", "J. Phys. Chem.", 1964, "68", "441-451",
               "10.1021/j100785a001")
R_GCVOL = _ref("Elbro, Fredenslund, Rasmussen", "Group contribution method for the prediction of liquid densities as "
               "a function of temperature for solvents, oligomers, and polymers", "Ind. Eng. Chem. Res.", 1991, "30",
               "2576-2582", "10.1021/ie00060a011")
R_BOSQUE = _ref("Bosque, Sales", "Polarizabilities of solvents from the chemical composition", "J. Chem. Inf. Comput. Sci.",
                2002, "42", "1154-1163", "10.1021/ci025528x")
R_CRIPPEN = _ref("Wildman, Crippen", "Prediction of physicochemical parameters by atomic contributions",
                 "J. Chem. Inf. Comput. Sci.", 1999, "39", "868-873", "10.1021/ci990307l")
R_MILLER = _ref("Miller", "Additivity methods in molecular polarizability", "J. Am. Chem. Soc.", 1990, "112",
                "8533-8542", "10.1021/ja00179a044")
R_D4 = _ref("Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher, Bannwarth, Grimme", "A generally applicable "
            "atomic-charge dependent London dispersion correction", "J. Chem. Phys.", 2019, "150", "154122",
            "10.1063/1.5090222")
R_SVPD = _ref("Rappoport, Furche", "Property-optimized Gaussian basis sets for molecular response calculations",
              "J. Chem. Phys.", 2010, "133", "134105", "10.1063/1.3484283")
R_SUGDEN = _ref("Sugden", "A relation between surface tension, density, and chemical composition",
                "J. Chem. Soc., Trans.", 1924, "125", "1177-1189", "10.1039/CT9242501177")
R_QUAYLE = _ref("Quayle", "The parachors of organic compounds. An interpretation and catalogue", "Chem. Rev.", 1953,
                "53", "439-589", "10.1021/cr60166a003")


# ============================================================================== entries
_FAMILY_ML, _FAMILY_TB, _FAMILY_NDDO, _FAMILY_FF, _FAMILY_CONF, _FAMILY_DFT, _FAMILY_BASIS, _FAMILY_SOLV, \
    _FAMILY_OPT, _FAMILY_SPARTAN, _FAMILY_EST, _FAMILY_CONCEPTS = FAMILIES

_HYBRID_COST = ("Hybrid DFT: minutes per conformer for PFAS. QuantaBench measured ωB97X-D/def2-SVP on PFHpA (23 atoms, "
                "313 basis functions): SCF 101 s; with properties, cubes and the polarizability 507 s (2 threads, "
                "docs/VALIDATION.md). Other hybrids cost roughly the same; time grows about as N^2.5 with the number "
                "of basis functions N.")
_MOPAC_COST = ("One MOPAC run (an external program) per energy and gradient. With Optimizer = engine's own, a whole "
               "optimization is one run, 3-15x faster than ASE stepping MOPAC (docs/ENGINES.md).")

_ENTRIES: tuple[Entry, ...] = (
    # ------------------------------------------------------------------ machine-learning potentials
    Entry(
        "aimnet2", "AIMNet2", _FAMILY_ML,
        "Neural network trained to reproduce ωB97M-D3(BJ)/def2-TZVPP DFT; 14 elements, closed shells, gas phase.",
        "A machine-learned interatomic potential: a neural network maps the atoms' positions and elements and the "
        "total charge to an energy, forces and atomic charges. It was trained on about 20 million hybrid-DFT "
        "calculations ([[wb97m|ωB97M]]/def2-TZVPP) of neutral and charged molecules. Long-range electrostatics and "
        "D3(BJ) dispersion are explicit physics terms added to the network's short-range energy.",
        "Conformer energies and geometries of organic molecules and ions made of H, B, C, N, O, F, Si, P, S, Cl, As, "
        "Se, Br and I, at a small fraction of the DFT cost. The first choice of `auto` when it is installed. The best "
        "cheap gas-phase conformer ranker QuantaBench tested on neutral PFAS: its lowest conformer was within 4 kJ/mol "
        "of ωB97X-D/6-31G*'s in 45 of 53 Essential 76 molecules (GFN2-xTB 36, a random pick 25; docs/VALIDATION.md, "
        "RANK-1).",
        "Only those 14 elements, no transition metals, closed shells only (radicals need AIMNet2-NSE, which "
        "QuantaBench does not use). Its developers validated it up to about 100 heavy atoms. No implicit solvent in "
        "QuantaBench, so a water run refines in the gas phase (with a note). No orbitals, so no HOMO/LUMO; the "
        "dipole comes from its predicted atomic charges. It ranks the conformers of PFAS ions in water only "
        "moderately, even with a solvation term from GFN2-xTB/CPCM added (median ρ 0.64 against ωB97X-D/def2-SVP in "
        "PCM over 17 ions, 0.50 on the 9 hardest; RANK-1), and leans toward extended conformers.",
        "Fastest on a GPU, which refines all conformers of a molecule in one batch: QuantaBench measured 8 PFOA "
        "conformers in 5.7 s on an RX 9060 XT and 59.8 s on the CPU (docs/PLATFORMS.md).",
        "Yes: D3 with Becke-Johnson damping (ωB97M parameters), added to the network energy.",
        "None (gas phase).",
        (R_AIMNET2,),
        aliases=("aimnet", "aimnet 2"),
        see_also=("wb97m", "dispersion", "auto"),
        implementations=(
            ("aimnet2", "aimnet package, in process. NVIDIA GPUs and the CPU through NVIDIA Warp; AMD (ROCm) and "
                        "Apple GPUs through QuantaBench's Warp-free path (`chem/aimnet_torch.py`), the same energies "
                        "to < 1e-5 eV. Batched refinement on a GPU."),
        ),
    ),
    # ------------------------------------------------------------------ tight binding
    Entry(
        "gfn2-xtb", "GFN2-xTB", _FAMILY_TB,
        "Grimme's extended tight-binding method, a fast approximation to DFT: good geometries and conformers, H to Rn.",
        "Tight binding is a semi-empirical simplification of DFT: a minimal valence basis, element parameters and "
        "approximate integrals. GFN stands for the targets of the fit: geometries, frequencies and noncovalent "
        "interactions. GFN2 adds atomic dipole and quadrupole electrostatics to GFN1's atomic charges and computes "
        "the charge-dependent [[dispersion|D4 dispersion]] self-consistently. It has parameters for every element "
        "up to radon (Z = 86).",
        "Geometries and conformer ranking of organic molecules, ions and organometallics, in the gas phase or in "
        "[[alpb|ALPB]] solvent, at low cost. Descriptors (dipole, charges, polarizability) for trends across "
        "molecules. QuantaBench's default refinement level for the PFAS validation.",
        "Orbital energies are not what it was fitted to: treat HOMO, LUMO and the gap as trend descriptors "
        "([[homo-lumo]]). QuantaBench measured, at the same geometries: gas-phase dipoles 0.81 D MAE from "
        "ωB97X-D/6-31G* (0.76 D higher on average, 63 PFAS), and water dipoles 0.52 D MAE from Spartan's "
        "B3LYP/C-PCM (61 ions; docs/VALIDATION.md, VAL-3). In the gas phase a multiply charged anion can have an "
        "unbound HOMO, and its polarizability can then diverge (QuantaBench discards implausible values with a note). "
        "Do not rank the conformers of PFAS ions with it, in any solvent model: against ωB97X-D/def2-SVP in PCM its "
        "lowest conformer was within 4 kJ/mol of DFT's in 4 of 17 ions with ALPB (5 with CPCM, 4 in the gas phase), "
        "no better than a random pick, and its median Spearman ρ was −0.13 (docs/VALIDATION.md, RANK-1). It puts "
        "conformers with short anionic O···F contacts (< 2.6 Å) 10-20 kJ/mol per contact too low, compactness held "
        "fixed. Its 6 kcal/mol window would have dropped DFT's lowest in 6 of the 17. Use it for geometries; rank "
        "ions' conformers with the QM stage in PCM (RANK-1 has a cheap alternative). For neutral PFAS in the gas "
        "phase its picks were within 4 kJ/mol of ωB97X-D/6-31G*'s in 36 of 53 molecules.",
        "Milliseconds to a fraction of a second per energy and gradient for PFAS-size molecules on one core. "
        "QuantaBench measured the whole pipeline for 24 PFAS on BTM NOVA: 126 s with tblite GFN2, against 76 s with "
        "AIMNet2 and 34 s with MMFF94s alone (docs/dev/BACKLOG.md, PERF-1).",
        "Yes: D4, charge-dependent and self-consistent.",
        "ALPB and GBSA (tblite, the xtb program, CREST); CPCM (tblite). None in dxtb.",
        (R_GFN2, R_XTB_REVIEW, R_DXTB),
        aliases=("gfn2", "gfn2 xtb", "xtb", "gfn2-xtb (dxtb)", "gfn2-xtb (tblite)", "gfn2-xtb (xtb program)"),
        see_also=("gfn1-xtb", "dispersion", "alpb", "homo-lumo", "crest-gfn2"),
        implementations=(
            ("gfn2-xtb", "dxtb (PyTorch, in process): autograd forces and Hessians, batched refinement. It can run on "
                         "a GPU but is 5-8x slower there than on the CPU (dxtb 0.4), so it stays on the CPU by "
                         "default. Gas phase only. Needs dxtb's libcint backend (the dxtb fork on Windows and macOS)."),
            ("tblite-gfn2", "tblite (the reference library, in process, CPU): ALPB, GBSA or CPCM solvent; dipole, "
                            "HOMO/LUMO, finite-field polarizability and the solvation energy. Matches the xtb "
                            "program to 4e-9 Eh."),
            ("xtb-gfn2", "the xtb program (one external run per calculation): ALPB or GBSA; its own optimizer, 4-10x "
                         "faster than ASE stepping xtb; its polarizability is the D4 model's α(0), not the SCF "
                         "response."),
        ),
    ),
    Entry(
        "gfn1-xtb", "GFN1-xTB", _FAMILY_TB,
        "The first GFN tight-binding method: like GFN2 with atomic charges only, D3(BJ) dispersion and a halogen term.",
        "GFN1-xTB (2017) is the first of the GFN tight-binding methods. Its electrostatics use atomic charges only "
        "(no atomic dipoles or quadrupoles), dispersion is [[dispersion|D3 with Becke-Johnson damping]], and a "
        "halogen-bond correction is added. It has parameters for H to Rn (Z = 1-86).",
        "A cross-check of GFN2-xTB results. QuantaBench's xTB properties step (the `xtb_*` charges, dipole and "
        "polarizability) uses it. dxtb runs it without the libcint backend, on the CPU or a GPU. With tblite's CPCM "
        "(ddX) its solvation energy differences between conformers of PFAS ions follow IEF-PCM's at ωB97X-D/def2-SVP "
        "to 3.9 kJ/mol MAE, closer than GFN2 with CPCM (9.7) or ALPB (20). Added to g-xTB's gas-phase energy, that "
        "term gave the cheap ranker closest to both DFT references tested (docs/VALIDATION.md, RANK-1).",
        "The same caveats as GFN2-xTB for orbital energies ([[homo-lumo]]) and for gas-phase anions. Its simpler "
        "electrostatics describe anisotropic contacts (hydrogen and halogen bonds) less well than GFN2's multipoles; "
        "GFN1 corrects halogen bonds with an extra term.",
        "About the same as GFN2-xTB.",
        "Yes: D3 with Becke-Johnson damping.",
        "ALPB and GBSA (tblite, the xtb program, CREST); CPCM (tblite). None in dxtb.",
        (R_GFN1, R_XTB_REVIEW),
        aliases=("gfn1", "gfn1 xtb", "gfn-xtb"),
        see_also=("gfn2-xtb", "dispersion", "crest-gfn1"),
        implementations=(
            ("gfn1-xtb", "dxtb (PyTorch, in process): CPU or GPU, batched refinement, gas phase. Agrees with "
                         "tblite's GFN1 to 5e-5 e in charges and 1e-4 D in dipoles (docs/ENGINES.md)."),
            ("tblite-gfn1", "tblite (in process, CPU): ALPB, GBSA or CPCM solvent; dipole, HOMO/LUMO and "
                            "finite-field polarizability."),
            ("xtb-gfn1", "the xtb program (external): ALPB or GBSA, its own optimizer."),
        ),
    ),
    Entry(
        "g-xtb", "g-xTB", _FAMILY_TB,
        "Grimme's general tight binding (2025, preliminary), fitted toward ωB97M-V; the rank step's gas-phase level.",
        "g-xTB (2025) is a tight-binding method from Grimme's group fitted toward ωB97M-V/def2-TZVPPD energies for the "
        "elements H to Lr (Z = 1-103). It keeps the minimal-basis tight-binding form of the GFN methods with new, "
        "charge-dependent terms. The official build of the xtb program runs it with `--gxtb` (`--chrg` and `--uhf` "
        "as for GFN2); its README calls it a preliminary version whose parameters may still change.",
        "The best cheap energy QuantaBench tested for ranking conformers. In the gas phase it picked a conformer within "
        "4 kJ/mol of r2SCAN-3c's lowest in 40 of 40 non-fluorinated neutrals and 18 of 20 neutral PFAS. With a "
        "GFN1-xTB CPCM solvation term at the same geometry it did so in water in 38 of 40 neutrals and 12 of 13 PFAS "
        "ions, and in 13 of 17 PFAS ions against ωB97X-D/def2-SVP (docs/VALIDATION.md, RANK-1 and RANK-1b).",
        "A rank level only (`--rank gxtb`, `gxtb+cpcm`, and `auto` in water), not a refinement engine: QuantaBench "
        "runs the official build of xtb (`xtb --gxtb`, its own program entry `gxtb`). v2.0.1 is a "
        "preliminary release whose parameters are still being optimized: QuantaBench records the build's SHA-256 "
        "with every molecule, and the installers pin it. Its own solvation models (generalized Born `--gbe` and COSMO) "
        "are provisional per its README and ranked worse than the GFN1-xTB CPCM term (RANK-1), so in water the rank "
        "step adds that term. The macOS arm64 build ships without libgcc_s.1.1.dylib (install_macos.sh supplies it).",
        "About 0.1 s per single point for 20-40 atoms on one thread (the program's start included); with the GFN1-xTB "
        "CPCM term 0.3-1.3 s per conformer (median 0.8), about 0.1 % of a ωB97X-D/def2-SVP single point for PFAS ions "
        "(QuantaBench measured, RANK-1, RANK-2).",
        "Yes, part of the method.",
        "Limited and provisional (generalized Born, COSMO); the rank step adds a GFN1-xTB CPCM term instead.",
        (R_GXTB,),
        aliases=("gxtb", "--gxtb", "g xtb", "gxtb+cpcm"),
        see_also=("gfn2-xtb", "gfn1-xtb", "conformer-ranking", "pcm"),
    ),
    # ------------------------------------------------------------------ semi-empirical NDDO
    Entry(
        "pm7", "PM7", _FAMILY_NDDO,
        "Stewart's NDDO semi-empirical method (MOPAC), fitted to experiment; dispersion and H-bond terms built in.",
        "NDDO methods (neglect of diatomic differential overlap) solve a minimal-basis Hartree-Fock problem in which "
        "most two-electron integrals are dropped and the rest come from fitted parameters. PM7 (2013) was fitted to "
        "experimental and high-level reference data, crystal structures included, with dispersion and hydrogen-bond "
        "terms in the method. MOPAC reports the energy as a heat of formation ΔHf.",
        "Heats of formation, dipoles and polarizabilities (MOPAC POLAR) of organic molecules; COSMO areas and "
        "volumes as size descriptors; a second semi-empirical opinion next to xTB. For simple gas-phase organic "
        "molecules Stewart reports average ΔHf errors about 10 % below PM6's.",
        "The energy is a heat of formation: compare it only within PM7. [[cosmo|COSMO]] in MOPAC is not smooth, so "
        "optimizations in water can stop short of the force threshold and COSMO frequencies are approximate. "
        "Elements: H to La and Lu to Bi. It ranks the conformers of PFAS ions as badly as GFN2-xTB does (median ρ "
        "against ωB97X-D/PCM over 17 ions −0.38 gas, −0.05 COSMO) and overbinds short anionic O···F contacts about "
        "as much (docs/VALIDATION.md, RANK-1).",
        _MOPAC_COST,
        "Yes: a damped pairwise dispersion term and a hydrogen-bond term, part of the method.",
        "COSMO (from the dielectric constant).",
        (R_PM7,),
        aliases=("pm7 (mopac)",),
        see_also=("pm6-d3h4x", "cosmo", "spartan-pm6"),
        implementations=(
            ("mopac-pm7", "MOPAC 22 or later (open source, external): COSMO solvent; heat of formation, dipole, "
                          "HOMO/LUMO, ionization energy, POLAR polarizability (closed shells)."),
        ),
    ),
    Entry(
        "pm6-d3h4x", "PM6-D3H4X", _FAMILY_NDDO,
        "PM6 plus Grimme's D3 dispersion and Řezáč-Hobza hydrogen-bond (H4) and halogen (X) corrections (MOPAC).",
        "PM6 (2007) is the NDDO method before PM7, parametrized for 70 elements. Plain PM6 describes dispersion and "
        "hydrogen bonds poorly. D3H4X adds Grimme's D3 dispersion, the H4 hydrogen-bond correction and a correction "
        "for halogen-oxygen and halogen-nitrogen contacts. QuantaBench's `mopac-pm6` engine always runs PM6-D3H4X.",
        "Noncovalent contacts (stacking, hydrogen and halogen bonds) at semi-empirical cost.",
        "Not the same as Spartan's PM6, which is plain PM6 ([[spartan-pm6]]): geometries and conformer energies of "
        "flexible molecules can differ. The energy is a heat of formation; COSMO caveats as for [[pm7|PM7]]. In the "
        "gas phase it overbinds short anionic O···F contacts as GFN2-xTB does; with COSMO it ranked PFAS ions' "
        "conformers better (median ρ 0.52 against ωB97X-D/PCM over 17 ions) but still picked one more than 4 kJ/mol "
        "above DFT's lowest in 4 of 17 (docs/VALIDATION.md, RANK-1).",
        _MOPAC_COST,
        "Yes: D3, plus the H4 hydrogen-bond and X halogen corrections.",
        "COSMO (from the dielectric constant).",
        (R_PM6, R_D3H4, R_D3H4X),
        aliases=("pm6", "pm6-d3h4", "pm6 d3h4x (mopac)"),
        see_also=("pm7", "spartan-pm6", "dispersion"),
        implementations=(
            ("mopac-pm6", "MOPAC 22 or later (open source, external): COSMO solvent; the same properties as PM7."),
        ),
    ),
    # ------------------------------------------------------------------ force fields
    Entry(
        "mmff94s", "MMFF94s", _FAMILY_FF,
        "Merck molecular force field (static variant): classical terms, no electrons; milliseconds, used for conformers.",
        "A force field computes the energy from bond stretches, angle bends, torsions, van der Waals contacts and "
        "fixed atomic charges, with no electrons. MMFF94 was parametrized on quantum-chemical data for organic and "
        "drug-like molecules; the 's' variant makes delocalized trigonal nitrogens (as in anilines) planar, which "
        "suits energy minimization. RDKit's MMFF94s minimizes every embedded conformer, and the method `mmff` stops "
        "there.",
        "Fast pre-optimization and a first ranking of many conformers; large batches where only geometries and 2D/3D "
        "shape descriptors are needed.",
        "Fixed charges: no polarization, no solvent, no electronic properties. Its conformer ranking is rough and in "
        "the gas phase, which is why QuantaBench refines the lowest 20 by default: as single points at fixed "
        "geometries its relative energies were 51 kJ/mol off ωB97X-D/PCM's on average for PFAS ions and 8 kJ/mol "
        "off ωB97X-D/6-31G*'s for neutral PFAS (docs/VALIDATION.md, RANK-1). Molecules it has no parameters for (Sn, Hg, B, hypervalent "
        "iodine, radicals) go to [[uff|UFF]].",
        "Milliseconds per conformer. QuantaBench measured 24 PFAS through the MMFF94s-only pipeline in 34 s on BTM "
        "NOVA (docs/dev/BACKLOG.md, PERF-1).",
        "A buffered 14-7 van der Waals term (part of the force field, not a DFT-style correction).",
        "None.",
        (R_MMFF94, R_MMFF94S),
        aliases=("mmff", "mmff94", "mmff94s (force field)"),
        see_also=("uff", "rdkit-etkdg"),
        implementations=(("mmff", "RDKit, in process, CPU threads; UFF for molecules MMFF94s has no parameters for."),),
    ),
    Entry(
        "uff", "UFF", _FAMILY_FF,
        "Universal force field: rule-based parameters for the whole periodic table; the fallback when MMFF94s has none.",
        "UFF builds its parameters from element and hybridization rules, so it covers the whole periodic table. "
        "QuantaBench (through RDKit) uses it for the conformer search when MMFF94s lacks parameters for a molecule, "
        "and says so in the notes and the `forcefield` column.",
        "Starting geometries for organometallics, tin and mercury compounds and other unusual valences.",
        "Less accurate than MMFF94s for organic molecules: treat its conformer energies only as a pre-sort before "
        "the refinement.",
        "Milliseconds per conformer.",
        "A Lennard-Jones van der Waals term.",
        "None.",
        (R_UFF,),
        aliases=("universal force field",),
        see_also=("mmff94s",),
    ),
    Entry(
        "gfn-ff", "GFN-FF", _FAMILY_FF,
        "Grimme's generic force field (H to Rn) that sets itself up from the structure; much faster than xTB.",
        "GFN-FF builds its own force field from the coordinates and elements: bond topology, bond orders and EEQ "
        "atomic charges come from the structure, so no atom types are needed. Dispersion is a simplified D4. On the "
        "conformational subsets of the GMTKN55 benchmark it has a mean absolute deviation of 1.53 kcal/mol, which "
        "its authors report as on par with the GFN-xTB methods.",
        "Large or unusual molecules, molecules with metals, and fast conformer searches ([[crest-gfnff]]).",
        "The bond topology is fixed when it is set up: bonds cannot form or break. The dipole comes from the EEQ "
        "charges, not from an electron density, and there are no orbitals. Its conformer energies are a search "
        "level, not a ranking: for neutral PFAS its picks were no better than random against ωB97X-D/6-31G* "
        "(within 4 kJ/mol in 22 of 53), and its errors for PFAS anions reached 100 kJ/mol (docs/VALIDATION.md, "
        "RANK-1).",
        "Far below xTB per step. CREST with GFN-FF searched 20 PFAS more than 6x faster than CREST with GFN2 on the "
        "same machine (QuantaBench measured, docs/ENGINES.md).",
        "Yes: a simplified D4.",
        "ALPB and GBSA (the xtb program, CREST).",
        (R_GFNFF,),
        aliases=("gfnff", "gff", "gfn-ff (xtb program)"),
        see_also=("crest-gfnff", "gfn2-xtb"),
        implementations=(("gfnff", "the xtb program (external): ALPB or GBSA, its own optimizer; EEQ charges and the "
                                   "charge dipole as properties."),),
    ),
    # ------------------------------------------------------------------ conformer search
    Entry(
        "rdkit-etkdg", "RDKit ETKDGv3 + MMFF94s", _FAMILY_CONF,
        "Distance-geometry embedding with experimental torsion preferences, then MMFF94s minimization; seconds per "
        "molecule.",
        "ETKDG builds 3D structures by distance geometry: random interatomic distances within bond-length and angle "
        "bounds, embedded in 3D and corrected with torsion-angle preferences from crystal structures and simple rules "
        "(flat aromatic rings, linear triple bonds). Version 3 improves small rings and macrocycles. QuantaBench embeds up to Max "
        "conformers structures (50 by default), minimizes each with [[mmff94s|MMFF94s]], and prunes by the energy "
        "window and an RMSD duplicate test.",
        "The default search: fast, reproducible with the random seed, and it needs no external program.",
        "The pool is ranked by gas-phase MMFF94s and only the lowest go on to the refinement, so conformers that a "
        "quantum method in water prefers can be lost there. In QuantaBench's comparison with Spartan (refining the "
        "5 lowest), Spartan's pool held a conformer GFN2/ALPB puts below QuantaBench's lowest for 49 of 61 ions "
        "(docs/VALIDATION.md, VAL-3). Raise Refine lowest for flexible ions, or use [[crest|CREST]].",
        "Seconds per molecule for PFAS. A cyclic peptide of 199 atoms took about 1.4 s per conformer on one core "
        "(docs/ENGINES.md).",
        "Only MMFF94s's van der Waals term.",
        "None: gas phase, the same pool in every phase.",
        (R_ETKDG, R_ETKDG3),
        aliases=("rdkit", "etkdg", "etkdgv3", "etkdg v3"),
        see_also=("mmff94s", "crest", "boltzmann"),
    ),
    Entry(
        "crest", "CREST iMTD-GC", _FAMILY_CONF,
        "Grimme's metadynamics conformer search on xTB or GFN-FF: finds conformers ETKDG misses; minutes to hours.",
        "CREST explores conformers with iterative metadynamics (iMTD): molecular dynamics with a bias that pushes "
        "the structure away from conformers it has already visited, measured by RMSD. Low structures are optimized, "
        "and a genetic crossover step (GC) combines parts of different conformers. Energies come from the chosen "
        "CREST level, in [[alpb|ALPB]] solvent when one is set. Quick mode (`--quick`, on by default here) uses "
        "reduced settings for a cruder ensemble; turn it off for CREST's full search.",
        "Flexible molecules and ions, especially in water, where a gas-phase force-field pool ranks poorly; "
        "ensembles for Boltzmann averages.",
        "Needs the crest program (Linux, the fork build on macOS, WSL on Windows). Run time grows quickly with size "
        "and flexibility. CREST can find tautomers or broken bonds: QuantaBench drops conformers whose bonds differ "
        "from the input. If CREST fails, the molecule gets RDKit's conformers (when that box is checked) and a note.",
        "GFN-FF `--quick`: ethanol about 2 s. GFN2 `--quick`: PFBA about 2 min on 2 cores, a C8 perfluoroalkyl acid "
        "tens of minutes (chem/crest.py).",
        "That of the CREST level: D4 (GFN2-xTB, GFN-FF) or D3(BJ) (GFN1-xTB).",
        "ALPB.",
        (R_CREST, R_CREST3),
        aliases=("crest imtd-gc", "imtd-gc", "imtd", "crest quick", "--quick"),
        see_also=("crest-gfn2", "crest-gfnff", "crest-gfn2//gfnff", "crest-gfn1", "rdkit-etkdg"),
    ),
    Entry(
        "crest-gfn2", "CREST level: GFN2-xTB", _FAMILY_CONF,
        "The whole CREST search at GFN2-xTB: CREST's default and most accurate level, and the slowest.",
        "Metadynamics, optimizations and the final ranking all use [[gfn2-xtb|GFN2-xTB]], in ALPB when a solvent "
        "is set.",
        "Final conformer ensembles when there is time, ions in water.",
        "Large flexible PFAS take hours: on the M3 Max (6 lanes x 2 threads) GFN2 took 20,034 s for 16 of 20 PFAS "
        "in the gas phase, and the 4 largest hit the 3 h limit per molecule (QuantaBench measured, docs/ENGINES.md).",
        "Minutes for small acids, hours for large flexible ones (above).",
        "Yes: D4.",
        "ALPB.",
        (R_CREST, R_GFN2),
        aliases=("crest gfn2", "crest --gfn2"),
        see_also=("crest", "gfn2-xtb"),
    ),
    Entry(
        "crest-gfnff", "CREST level: GFN-FF", _FAMILY_CONF,
        "The whole search with the GFN-FF force field: more than 6x faster than GFN2; ranking at force-field level.",
        "Metadynamics, optimizations and the ranking use [[gfn-ff|GFN-FF]]. The Refinement stage then re-optimizes "
        "the lowest conformers with the refinement method.",
        "Large molecules and big batches; a first ensemble that the refinement re-ranks.",
        "The ranking is at force-field level, so refine with xTB or better (the pipeline does). Bonds are fixed by "
        "GFN-FF's topology.",
        "QuantaBench measured 20 PFAS in the gas phase: 3,108 s on 6 lanes x 2 threads and 773 s on 3 x 4 threads "
        "(M3 Max), 1,247 s on BTM NOVA's WSL (20 x 2) (docs/ENGINES.md).",
        "Yes: a simplified D4.",
        "ALPB.",
        (R_CREST, R_GFNFF),
        aliases=("crest gfnff", "crest gff", "crest --gfnff"),
        see_also=("crest", "gfn-ff"),
    ),
    Entry(
        "crest-gfn2//gfnff", "CREST level: GFN2-xTB // GFN-FF", _FAMILY_CONF,
        "Samples and optimizes with GFN-FF, ranks with GFN2-xTB single points; needs the xtb program.",
        "'A//B' means energies at A on geometries from B. CREST runs the metadynamics and optimizations with "
        "[[gfn-ff|GFN-FF]] and evaluates the energies and sorts the ensemble with [[gfn2-xtb|GFN2-xTB]]. Its "
        "accuracy lies between the two levels, because the geometries keep GFN-FF's biases (CREST's developers, "
        "crest-lab/crest issue 72).",
        "Big searches that should be ranked at a quantum level.",
        "CREST 3.0.2 runs this level only in `--legacy` mode, which calls the xtb program for every calculation, so "
        "it needs xtb and is slower than CREST's built-in GFN-FF.",
        "Between GFN-FF and GFN2-xTB.",
        "Yes: D4 (both levels).",
        "ALPB.",
        (R_CREST,),
        aliases=("gfn2//gfnff", "crest gfn2//gfnff", "gfn2/gfnff"),
        see_also=("crest", "single-point", "gfn-ff", "gfn2-xtb"),
    ),
    Entry(
        "crest-gfn1", "CREST level: GFN1-xTB", _FAMILY_CONF,
        "The whole CREST search at GFN1-xTB: the other tight-binding level, at about GFN2's cost.",
        "Metadynamics, optimizations and the ranking use [[gfn1-xtb|GFN1-xTB]].",
        "A cross-check of a GFN2 ensemble.",
        "The same run times as GFN2; GFN1's simpler electrostatics rank polar conformers differently.",
        "About that of CREST with GFN2-xTB.",
        "Yes: D3(BJ).",
        "ALPB.",
        (R_CREST, R_GFN1),
        aliases=("crest gfn1", "crest --gfn1"),
        see_also=("crest", "gfn1-xtb"),
    ),
    # ------------------------------------------------------------------ DFT
    Entry(
        "dft", "DFT and functional names", _FAMILY_DFT,
        "Density functional theory gets the energy from the electron density; the functional's name says which "
        "approximation.",
        "DFT computes the energy from the electron density. The unknown exchange-correlation part is approximated by "
        "a functional. From cheap to expensive: GGA (density and its gradient: PBE, BLYP), meta-GGA (also the "
        "kinetic-energy density: r2SCAN), hybrid (a fixed share of exact Hartree-Fock exchange: B3LYP 20 %, PBE0 "
        "25 %, M06-2X 54 %) and range-separated hybrid (the share of exact exchange grows with the distance between "
        "electrons: ωB97X-D). Suffixes name the [[dispersion|dispersion correction]] (-D3BJ, -D4, -D, -V).",
        "Reading level names: 'ωB97X-D/6-31G*' is functional/[[basis-sets|basis set]]; "
        "'B3LYP/6-311+G**//MMFF' is a B3LYP single point at an MMFF geometry ([[single-point]]).",
        "A level is functional + basis set + solvent model + geometry. The same functional with another basis, "
        "solvent model or geometry gives other numbers. GMTKN55, a broad benchmark of main-group chemistry, "
        "recommends dispersion corrections throughout and does not include B3LYP in its recommendations.",
        _HYBRID_COST,
        "Depends on the functional (see each entry).",
        "PCM in Psi4 (single points).",
        (R_PSI4, R_GMTKN55),
        aliases=("density functional theory", "functional", "functionals", "qm", "psi4"),
        see_also=("wb97x-d", "b3lyp-d3bj", "basis-sets", "dispersion"),
    ),
    Entry(
        "wb97x-d", "ωB97X-D", _FAMILY_DFT,
        "Range-separated hybrid with its own dispersion term (Chai and Head-Gordon); QuantaBench's default functional.",
        "A range-separated hybrid: about 22 % exact exchange between nearby electrons, rising to 100 % at long "
        "range (switching length set by ω = 0.2 per bohr). The '-D' is its own damped atom-pair dispersion term, "
        "fitted together with the functional; it is an older pairwise form, not Grimme's D3. Full exact exchange at "
        "long range improves anions, charge transfer and orbital energies over global hybrids such as B3LYP.",
        "General organic chemistry: conformer energies, noncovalent contacts, dipoles and polarizabilities. Matching "
        "Spartan's ωB97X-D/6-31G* results: at Spartan's geometry QuantaBench's Psi4 stage reproduces its energies "
        "to 4e-4 Eh and its dipoles to 0.001 D (docs/VALIDATION.md, VAL-2).",
        "Its dispersion term predates D3(BJ) and D4. Psi4 computes its polarizability analytically, except in PCM.",
        _HYBRID_COST,
        "Yes: its own damped atom-pair term ('-D').",
        "PCM in Psi4 (single points).",
        (R_WB97XD,),
        aliases=("wb97xd", "ωb97xd"),
        see_also=("dispersion", "spartan-wb97x-d", "b3lyp-d3bj", "dft"),
    ),
    Entry(
        "b3lyp-d3bj", "B3LYP-D3(BJ)", _FAMILY_DFT,
        "The classic global hybrid (20 % exact exchange) with Grimme's D3 dispersion and Becke-Johnson damping.",
        "B3LYP mixes 20 % exact (Hartree-Fock) exchange with Becke's gradient-corrected exchange and the "
        "Lee-Yang-Parr correlation functional, with three fitted mixing parameters. The -D3(BJ) suffix adds Grimme's "
        "D3 dispersion with Becke-Johnson damping; without it B3LYP misses most of the London dispersion between "
        "nonbonded atoms ([[b3lyp]]).",
        "Comparing with the large literature that uses B3LYP, with the dispersion fixed; geometries and vibrational "
        "frequencies. Ranking the conformers of PFAS ions in PCM water: at def2-SVP it agreed with ωB97X-D (median ρ "
        "0.95; its lowest within 4 kJ/mol of ωB97X-D's in 9 of 9 ions) at 0.72x the cost (docs/VALIDATION.md, "
        "RANK-1).",
        "A global hybrid: less suited to anions and charge transfer than range-separated ones. GMTKN55 does not "
        "include B3LYP in its recommendations. Needs `dftd3-python` in the Psi4 environment. Its agreement with "
        "ωB97X-D/def2-SVP on PFAS ions comes largely from the shared basis: against r2SCAN-3c in PCM, def2-SVP "
        "B3LYP-D3(BJ) favoured compact conformers (median ρ 0.60, 3 of 5 picks within 4 kJ/mol, against 0.94 and 5 of "
        "5 for g-xTB + GFN1-xTB CPCM; docs/VALIDATION.md, RANK-1).",
        _HYBRID_COST + " D3 itself adds almost nothing.",
        "Yes: D3 with Becke-Johnson damping.",
        "PCM in Psi4 (single points).",
        (R_B3, R_B3LYP, R_D3BJ),
        aliases=("b3lyp-d3(bj)", "b3lyp-d3", "b3lypd3bj"),
        see_also=("b3lyp", "dispersion", "pbe0-d3bj", "dft"),
    ),
    Entry(
        "b3lyp", "B3LYP (no dispersion)", _FAMILY_DFT,
        "B3LYP without a dispersion correction, as in Spartan's B3LYP runs: underbinds folded and stacked structures.",
        "Plain B3LYP is the functional of many older studies and of Spartan's B3LYP/6-311+G** calculations. It has "
        "no dispersion correction. London dispersion is the attraction between atoms that are close but not bonded, "
        "so a folded conformer, where chain segments touch, loses an attraction an extended one does not need. "
        "Plain B3LYP therefore tends to favour extended chains over folded ones. Even n-alkanes longer than about 17 "
        "carbons fold into hairpins in the gas phase, held together by dispersion between the strands.",
        "Reproducing older or Spartan results. Type `b3lyp` in the Functional box; use b3lyp-d3bj for new work.",
        "This matters for PFAS conformers. In QuantaBench's comparison, GFN2/ALPB (with D4) put compact, low-dipole "
        "ions lowest and Spartan's B3LYP/C-PCM single points put extended ones lowest, and the conformer choice is "
        "88 % of the 4.35 D water-dipole gap (docs/VALIDATION.md, VAL-3). Dispersion is only a small part of that: "
        "at common GFN2 geometries D3(BJ) moves the energy difference between the two picks by about 10 kJ/mol, and "
        "ωB97X-D and B3LYP-D3(BJ) in PCM water side with Spartan's pick, not GFN2's (the DFT conformer test, VAL-3).",
        _HYBRID_COST,
        "No.",
        "PCM in Psi4 (single points).",
        (R_B3LYP, R_DISP_REVIEW, R_ALKANE),
        aliases=("b3lyp plain", "b3lyp without dispersion"),
        see_also=("b3lyp-d3bj", "dispersion", "spartan-b3lyp"),
    ),
    Entry(
        "pbe0-d3bj", "PBE0-D3(BJ)", _FAMILY_DFT,
        "Non-empirical global hybrid (PBE with 25 % exact exchange) plus D3(BJ) dispersion: a general-purpose choice.",
        "PBE0 takes the PBE gradient-corrected functional, built from physical constraints rather than fitted to "
        "data, and replaces 25 % of its exchange with exact exchange. D3(BJ) adds dispersion.",
        "Geometries, dipoles and polarizabilities; a second hybrid to check B3LYP or ωB97X-D results.",
        "A global hybrid, with the same long-range limits as B3LYP for anions and charge transfer.",
        _HYBRID_COST,
        "Yes: D3 with Becke-Johnson damping.",
        "PCM in Psi4 (single points).",
        (R_PBE, R_PBE0, R_D3BJ),
        aliases=("pbe0-d3(bj)", "pbe0", "pbeh", "pbe0-d3"),
        see_also=("b3lyp-d3bj", "dispersion", "dft"),
    ),
    Entry(
        "r2scan", "r2SCAN", _FAMILY_DFT,
        "Meta-GGA without exact exchange: cheaper than hybrids; no long-range dispersion as listed (r2SCAN-D4 adds it).",
        "A meta-GGA uses the kinetic-energy density as well as the density and its gradient, which lets it tell "
        "single bonds, double bonds and weak contacts apart. r2SCAN (regularized-restored SCAN) keeps most of SCAN's "
        "exact constraints and is numerically stable on ordinary integration grids. It has no exact exchange.",
        "Geometries and conformer energies of larger molecules when hybrids are too slow. The composite r2SCAN-3c "
        "was QuantaBench's check on the conformer reference for PFAS ions in PCM water. On 5 ions it agreed with "
        "ωB97X-D/def2-SVPD (median ρ 0.99) better than ωB97X-D/def2-SVP did (0.72). Of the cheap rankers, it agreed "
        "best with g-xTB + GFN1-xTB CPCM (ρ 0.94; docs/VALIDATION.md, RANK-1).",
        "It captures some medium-range dispersion but not long-range dispersion: r2SCAN-D4 adds it (Psi4 needs "
        "`dftd4-python` for -d4 names). Without exact exchange, HOMO-LUMO gaps come out smaller than with hybrids. "
        "Psi4 uses a finite-field polarizability for meta-GGAs.",
        "No exact exchange: with density fitting, cheaper than a hybrid in the same basis. Not so the composite "
        "r2SCAN-3c (def2-mTZVPP basis, D4, gCP; Psi4 needs `mctc-gcp` on the PATH): in PCM it took 2x "
        "ωB97X-D/def2-SVP for a 37-atom PFAS ion in Psi4 1.9.1 (QuantaBench measured, RANK-1).",
        "Medium range only, from the functional itself.",
        "PCM in Psi4 (single points).",
        (R_R2SCAN, R_R2SCAN_D4, R_R2SCAN3C),
        aliases=("r2-scan", "r²scan"),
        see_also=("r2scan-3c", "dispersion", "dft"),
    ),
    Entry(
        "r2scan-3c", "r2SCAN-3c and the -3c composites", _FAMILY_DFT,
        "Grimme's composite DFT recipes: functional, matched basis, D4 or D3 and gCP in one; RANK-1b's reference.",
        "A '-3c' method is a recipe of three parts chosen together: a functional, a basis set made for it, and "
        "corrections for the two errors that basis leaves, [[dispersion|D4 or D3 dispersion]] and the "
        "[[gcp|gCP]] correction for basis-set superposition error. r2SCAN-3c uses the [[r2scan|r2SCAN]] meta-GGA "
        "with the triple-zeta def2-mTZVPP basis and D4; HF-3c (Hartree-Fock, the minimal MINIX basis, D3(BJ)) and "
        "PBEh-3c are cheaper relatives. Psi4 runs them by name with their own basis (`energy('r2scan3c/')`).",
        "Conformer energies near triple-zeta hybrid quality at about twice the cost of ωB97X-D/def2-SVP. QuantaBench "
        "used r2SCAN-3c as its conformer reference: it agreed with ωB97X-D/def2-TZVP on 8 neutrals (ρ 0.98, MAE "
        "1.1 kJ/mol) and, on PFAS ions in PCM, with ωB97X-D/def2-SVPD better than ωB97X-D/def2-SVP did "
        "(docs/VALIDATION.md, RANK-1 and RANK-1b).",
        "Not selectable in the QM stage yet: the stage takes 'functional/basis', and Psi4 wants the -3c methods with "
        "an empty basis. Psi4 needs `mctc-gcp` on the PATH and `dftd4-python` for r2SCAN-3c. HF-3c is much cheaper "
        "but ranked PFAS conformers poorly (10 of 17 ions within 4 kJ/mol, leaning to extended ones; RANK-1).",
        "r2SCAN-3c in PCM: 2x ωB97X-D/def2-SVP for a 37-atom PFAS ion on a 4-thread NOVA lane; HF-3c 0.2-0.3x "
        "(QuantaBench measured, Psi4 1.9.1, RANK-1).",
        "Yes: D4 (r2SCAN-3c), D3 with Becke-Johnson damping (HF-3c, PBEh-3c).",
        "PCM in Psi4 (single points).",
        (R_R2SCAN3C, R_HF3C, R_GCP),
        aliases=("r2scan3c", "hf-3c", "hf3c", "pbeh-3c", "pbeh3c", "3c", "composite methods"),
        see_also=("gcp", "r2scan", "dispersion", "conformer-ranking"),
    ),
    Entry(
        "m06-2x", "M06-2X", _FAMILY_DFT,
        "Truhlar's highly parametrized hybrid meta-GGA (54 % exact exchange) for main-group thermochemistry, barriers.",
        "M06-2X is one of the Minnesota 2006 functionals; '2X' means twice the exact exchange of M06 (54 %). It was "
        "fitted to main-group thermochemistry, reaction barriers and noncovalent interactions, so its parameters "
        "describe medium-range dispersion.",
        "Reaction energies and barriers of organic molecules; noncovalent complexes at short range.",
        "No long-range dispersion: GMTKN55 found that Minnesota functionals still need a D3 correction. Meta-GGAs "
        "of the M06 family are sensitive to the integration grid. Psi4 uses a finite-field polarizability.",
        _HYBRID_COST + " It needs a finer integration grid for stable results.",
        "Medium range only, through its parametrization.",
        "PCM in Psi4 (single points).",
        (R_M062X, R_GMTKN55, R_GRID),
        aliases=("m062x", "m06 2x"),
        see_also=("dispersion", "dft"),
    ),
    Entry(
        "hf", "Hartree-Fock (HF)", _FAMILY_DFT,
        "Mean-field wavefunction theory without electron correlation: a baseline, not a level for production energies.",
        "Each electron moves in the average field of the others. Exchange is exact, but electron correlation is "
        "missing entirely, dispersion included. Correlated wavefunction methods start from HF, and the NDDO methods "
        "(PM6, PM7) are parametrized HF.",
        "A baseline; Koopmans' estimate of the ionization energy (−ε_HOMO); the HF/6-31G* electrostatic potential "
        "that classic force-field charges are fitted to.",
        "No correlation and no dispersion: conformer energies of flexible chains and noncovalent contacts are "
        "unreliable, and HOMO-LUMO gaps are much larger than DFT's. HF-3c (Psi4's composite: the MINIX basis, D3, "
        "gCP and a short-range basis correction; it needs `mctc-gcp` on the PATH) does not fix that for PFAS "
        "conformers: against ωB97X-D/def2-SVP in PCM its lowest conformer was within 4 kJ/mol in 10 of 17 ions, "
        "favouring extended ones, and against ωB97X-D/6-31G* in 18 of 27 neutrals (docs/VALIDATION.md, RANK-1).",
        "Similar to a hybrid DFT calculation in the same basis. HF-3c: 0.2-0.3x ωB97X-D/def2-SVP for PFAS ions in "
        "PCM, where the PCM steps then dominate (QuantaBench measured, RANK-1).",
        "No.",
        "PCM in Psi4 (single points).",
        (R_ROOTHAAN, R_KOOPMANS, R_HF3C),
        aliases=("hartree-fock", "rhf", "uhf", "scf"),
        see_also=("dft", "homo-lumo", "r2scan-3c"),
    ),
    Entry(
        "wb97m", "ωB97M-D3(BJ) / ωB97M-V", _FAMILY_DFT,
        "Head-Gordon's range-separated hybrid meta-GGA; AIMNet2 was trained on ωB97M with D3(BJ) dispersion.",
        "ωB97M-V combines range-separated exact exchange (as in ωB97X-D) with a meta-GGA form and the VV10 nonlocal "
        "correlation functional for dispersion; its terms were chosen by a combinatorial search over many forms. "
        "The D3(BJ) variant uses D3(BJ) instead of VV10. AIMNet2's training data are ωB97M/def2-TZVPP with D3(BJ), "
        "so AIMNet2's numbers aim at this level, not at B3LYP or ωB97X-D.",
        "Knowing what [[aimnet2|AIMNet2]] imitates. Psi4 runs ωB97M-V when you type `wb97m-v` in the Functional box.",
        "VV10 makes Psi4 use a finite-field polarizability and adds grid work.",
        _HYBRID_COST,
        "Yes: VV10 (ωB97M-V) or D3(BJ).",
        "PCM in Psi4 (single points).",
        (R_WB97MV, R_AIMNET2),
        aliases=("wb97m-v", "wb97m-d3bj", "wb97m-d3(bj)", "wb97m-d3", "ωb97m-v"),
        see_also=("aimnet2", "wb97x-d", "dispersion"),
    ),
    # ------------------------------------------------------------------ basis sets
    Entry(
        "basis-sets", "Basis sets (what the names mean)", _FAMILY_BASIS,
        "The functions molecular orbitals are built from: more and wider functions give better answers at a steep cost.",
        "Each molecular orbital is a sum of atom-centred Gaussian functions. Split-valence bases ('SV', 6-31G) give "
        "each valence orbital two functions of different size, triple-zeta bases ('TZV', 6-311G) three. "
        "Polarization functions ('P', '*', '(d,p)') add higher angular momentum (d on C, N, O, F; p on H) so "
        "orbitals can bend and bonds can polarize. Diffuse functions ('D', '+', 'aug-') are wide, slowly decaying "
        "functions for electrons far from the nuclei.",
        "def2-SVP for screening, def2-TZVP for final energies, a diffuse basis (def2-SVPD, 6-31+G*, aug-cc-pVDZ) "
        "for anions, polarizabilities and ESP surfaces.",
        "Anions need diffuse functions: the extra electron occupies a spread-out orbital that a compact basis "
        "squeezes, which raises the anion's energy and shrinks its polarizability (carboxylates, sulfonates). "
        "Psi4 and Spartan use 6 Cartesian d functions in 6-31G* (QuantaBench checked the basis-function counts, "
        "docs/VALIDATION.md); def2 and Dunning sets use 5 spherical ones.",
        "Basis functions for PFOA (C8HF15O2, 26 atoms): def2-SVP 355, 6-31G* 377, 6-31+G* 477, 6-311+G** 556, "
        "def2-SVPD 559, aug-cc-pVDZ 584, def2-TZVP 781 (6 Cartesian d functions in 6-31G* and 6-31+G*, 5 spherical "
        "ones in the others). QuantaBench's Psi4 time grows about as N^2.5 (docs/VALIDATION.md), so def2-TZVP costs "
        "about 7x def2-SVP.",
        "Not applicable (a basis set does not add dispersion).",
        "Not applicable.",
        (R_DEF2, R_631G, R_DIFFUSE),
        aliases=("basis", "basis set", "diffuse functions", "polarization functions", "split valence"),
        see_also=("def2-svp", "def2-tzvp", "def2-svpd", "6-31g*", "aug-cc-pvdz"),
    ),
    Entry(
        "def2-svp", "def2-SVP", _FAMILY_BASIS,
        "Ahlrichs split-valence double-zeta with polarization: the standard fast DFT basis (355 functions for PFOA).",
        "Two functions per valence orbital, one d set on heavy atoms and one p set on hydrogen. The def2 family "
        "(Weigend and Ahlrichs) is balanced across the periodic table, H to Rn, with effective core potentials "
        "beyond Kr.",
        "Geometries and screening at hybrid-DFT level; QuantaBench's default QM basis.",
        "No diffuse functions, so anions come out too compact. QuantaBench measured, against ωB97X-D/6-31G* at the "
        "same geometry for TFA and PFPrA: dipoles −0.05 to −0.07 D, HOMO/LUMO −0.21 to −0.25 eV, polarizability "
        "−4 % (docs/VALIDATION.md, VAL-2). For conformers of PFAS ions in PCM water, ωB97X-D/def2-SVP favoured compact "
        "conformers over extended ones by 11-13 kJ/mol on average relative to def2-SVPD and r2SCAN-3c "
        "(docs/VALIDATION.md, RANK-1): use a diffuse or triple-zeta basis to rank them.",
        "PFHpA (313 functions) at ωB97X-D: SCF 101 s, full protocol 507 s on 2 threads (QuantaBench measured).",
        "Not applicable.",
        "Not applicable.",
        (R_DEF2,),
        aliases=("def2svp", "svp"),
        see_also=("basis-sets", "def2-svpd", "def2-tzvp"),
    ),
    Entry(
        "def2-svpd", "def2-SVPD", _FAMILY_BASIS,
        "def2-SVP plus diffuse functions tuned for response properties: for anions and polarizabilities.",
        "def2-SVP with diffuse functions (Rappoport and Furche) chosen to give accurate polarizabilities and other "
        "response properties at small cost.",
        "Anions (carboxylates, sulfonates), polarizabilities and ESP surfaces at DFT level. Conformer energies of "
        "PFAS ions: on the ion where ωB97X-D/def2-SVP and r2SCAN-3c disagreed most, def2-SVPD moved ωB97X-D to "
        "within 8 kJ/mol MAE of r2SCAN-3c, from 16 (docs/VALIDATION.md, RANK-1).",
        "Diffuse functions slow the SCF and can make it harder to converge.",
        "559 functions for PFOA: about 3x the def2-SVP time. ωB97X-D/def2-SVPD in PCM took 740-850 s for a 37-atom "
        "PFAS ion on 4 threads, about 3x def2-SVP (QuantaBench measured, RANK-1).",
        "Not applicable.",
        "Not applicable.",
        (R_DEF2D,),
        aliases=("def2svpd", "svpd"),
        see_also=("basis-sets", "def2-svp"),
    ),
    Entry(
        "def2-tzvp", "def2-TZVP", _FAMILY_BASIS,
        "Ahlrichs triple-zeta with polarization: close to converged DFT energies; about 7x the cost of def2-SVP.",
        "Three functions per valence orbital, two d sets and one f set on heavy atoms (C to F) and one p set on "
        "hydrogen.",
        "Final energies and properties of neutral molecules at DFT level.",
        "No diffuse functions, so anions and polarizabilities still need a diffuse basis.",
        "781 functions for PFOA: about 7x the def2-SVP time.",
        "Not applicable.",
        "Not applicable.",
        (R_DEF2,),
        aliases=("def2tzvp", "tzvp"),
        see_also=("basis-sets", "def2-svp"),
    ),
    Entry(
        "6-31g*", "6-31G*", _FAMILY_BASIS,
        "Pople split-valence with d functions on heavy atoms: the classic basis, and Spartan's ωB97X-D/6-31G* level.",
        "Core orbitals use one function (a contraction of 6 Gaussians), valence orbitals two (3 + 1 Gaussians); '*' "
        "adds one set of d functions on non-hydrogen atoms. Also written 6-31G(d).",
        "Like-for-like comparisons with Spartan's ωB97X-D/6-31G* results; quick DFT geometries.",
        "No polarization on hydrogen (6-31G** adds it) and no diffuse functions for anions.",
        "377 functions for PFOA. PFHpA (332) at ωB97X-D: SCF 127 s, full protocol 533 s on 2 threads (QuantaBench "
        "measured).",
        "Not applicable.",
        "Not applicable.",
        (R_631G, R_POLARIZATION),
        aliases=("6-31g(d)", "631g*", "6-31gs"),
        see_also=("basis-sets", "6-31+g*", "spartan-wb97x-d"),
    ),
    Entry(
        "6-31+g*", "6-31+G*", _FAMILY_BASIS,
        "6-31G* plus diffuse s and p functions on heavy atoms: the cheapest Pople basis for anions.",
        "'+' adds one diffuse s and one diffuse p function on every non-hydrogen atom. Also written 6-31+G(d).",
        "Anions at modest cost.",
        "Still no functions beyond s on hydrogen.",
        "477 functions for PFOA: about 2x the def2-SVP time.",
        "Not applicable.",
        "Not applicable.",
        (R_631G, R_DIFFUSE),
        aliases=("6-31+g(d)", "631+g*"),
        see_also=("basis-sets", "6-31g*"),
    ),
    Entry(
        "6-311+g**", "6-311+G**", _FAMILY_BASIS,
        "Pople triple-split valence with diffuse and polarization functions: the basis of Spartan's B3LYP runs.",
        "Valence orbitals get three functions; '+' adds diffuse s and p on heavy atoms and '**' adds d on heavy atoms "
        "and p on hydrogen. Also written 6-311+G(d,p).",
        "Like-for-like comparisons with Spartan's B3LYP/6-311+G** water results; anions.",
        "No f functions, and it costs about as much as def2-SVPD.",
        "556 functions for PFOA: about 3x the def2-SVP time.",
        "Not applicable.",
        "Not applicable.",
        (R_6311G, R_DIFFUSE),
        aliases=("6-311+g(d,p)", "6311+g**"),
        see_also=("basis-sets", "spartan-b3lyp"),
    ),
    Entry(
        "aug-cc-pvdz", "aug-cc-pVDZ", _FAMILY_BASIS,
        "Dunning correlation-consistent double-zeta with diffuse functions on every atom, hydrogen included.",
        "cc-pVnZ sets were designed so that results converge systematically from DZ to TZ to QZ in correlated "
        "calculations. 'aug-' adds one diffuse function of each angular momentum on every atom.",
        "Anions, polarizabilities and weak complexes; the first step of a basis-set convergence series.",
        "Diffuse functions on all atoms: slower, and in large molecules nearly linearly dependent functions can stall "
        "the SCF. For DFT, def2-SVPD costs about the same and was tuned for response properties.",
        "584 functions for PFOA: about 3.5x the def2-SVP time.",
        "Not applicable.",
        "Not applicable.",
        (R_CCPVDZ, R_AUG),
        aliases=("augccpvdz", "aug-cc-pvdz (dunning)"),
        see_also=("basis-sets", "def2-svpd"),
    ),
    # ------------------------------------------------------------------ implicit solvation
    Entry(
        "solvation", "Implicit solvation", _FAMILY_SOLV,
        "The solvent as a polarizable continuum around the molecule; one name ('water') maps onto each engine's model.",
        "An implicit (continuum) model replaces the solvent molecules with a dielectric medium outside a cavity "
        "around the solute. The solute's charges polarize the medium, which acts back on them (the reaction field); "
        "some models add empirical terms for cavity formation and dispersion with the solvent. QuantaBench takes one "
        "solvent name and maps it onto each engine's model: [[alpb|ALPB]] in tblite, xtb and CREST, [[cosmo|COSMO]] "
        "(from ε) in MOPAC, [[pcm|PCM]] in Psi4. Engines without one (dxtb, AIMNet2, MMFF94s) run in the gas phase "
        "and the notes say so.",
        "Ions and polar molecules, where water changes conformer rankings and raises dipoles: at the same geometry "
        "PCM water raised the ωB97X-D/6-31G* dipoles of TFA and PFPrA by 26-28 % (QuantaBench measured, "
        "docs/VALIDATION.md).",
        "No explicit hydrogen bonds to water and no counter-ions. The RDKit conformer search is always in the gas "
        "phase; only CREST searches in ALPB. Solvation energies from different models are not interchangeable.",
        "ALPB adds little to an xTB calculation. In Psi4, PCM rules out analytic gradients and polarizabilities "
        "(gas-phase optimizations, finite fields).",
        "Not applicable.",
        "The subject of this family.",
        (R_IMPLICIT_REVIEW, R_PCM_REVIEW),
        aliases=("solvent", "implicit solvent", "continuum solvation", "water"),
        see_also=("alpb", "gbsa", "cosmo", "pcm", "smd", "gas-phase"),
    ),
    Entry(
        "gas-phase", "Gas phase", _FAMILY_SOLV,
        "The isolated molecule in vacuum: no solvent, and nothing screens an ion's charge.",
        "With no solvent, or with an engine that has no solvent model, every step treats the molecule as isolated. "
        "This is the reference state of most benchmark data and of Spartan's PM6 and ωB97X-D/6-31G* results.",
        "Neutral molecules; comparison with gas-phase reference data.",
        "Ions are unscreened: a multiply charged anion can be unbound in the gas phase (its HOMO above zero), and its "
        "xTB polarizability can then diverge. QuantaBench reports ions' dipoles about the centre of mass, because an ion's "
        "dipole depends on the origin.",
        "None.",
        "Not applicable.",
        "None.",
        (),
        aliases=("gas", "vacuum", "none"),
        see_also=("solvation",),
    ),
    Entry(
        "alpb", "ALPB", _FAMILY_SOLV,
        "Analytical linearized Poisson-Boltzmann: the xTB solvent model (tblite, xtb, CREST), fitted per solvent.",
        "ALPB is a generalized-Born-type model: it estimates the continuum's reaction field from atomic Born radii, "
        "with an analytical correction for the solute's shape, and adds a surface-area term for cavity and "
        "dispersion effects. It is parametrized for each method (GFN1, GFN2, GFN-FF) and for more than 20 solvents. "
        "Its authors report hydration free energies with GFN1-xTB(ALPB) within 1.4 kcal/mol of experiment on "
        "average, and octanol-water log K within about 0.65 with GFN2-xTB(ALPB).",
        "Fast solvated optimizations and conformer ranking at xTB level; QuantaBench's water runs.",
        "A continuum: no explicit hydrogen bonds to water. GFN2/ALPB and B3LYP/C-PCM rank folded and extended PFAS "
        "ions differently (VAL-3). Between conformers of one PFAS ion, ALPB's solvation energy differences missed "
        "IEF-PCM's (at ωB97X-D/def2-SVP) by 20 kJ/mol on average, favouring the compact conformer; tblite's CPCM "
        "with GFN1-xTB missed them by 4 (docs/VALIDATION.md, RANK-1). ALPB's non-polar terms are not the cause: its "
        "electrostatic term alone does no better.",
        "Little more than the gas-phase xTB calculation.",
        "Not applicable.",
        "Yes: more than 20 solvents.",
        (R_ALPB,),
        aliases=("alpb solvation", "--alpb"),
        see_also=("gbsa", "solvation", "gfn2-xtb"),
    ),
    Entry(
        "gbsa", "GBSA", _FAMILY_SOLV,
        "Generalized Born + solvent-accessible surface area: the older xTB solvent model, which ALPB replaced.",
        "Generalized Born estimates the electrostatic solvation energy from pairwise atomic terms with effective "
        "Born radii; the SASA term adds cavity and dispersion effects in proportion to the exposed surface. The "
        "CREST documentation calls ALPB an improved version of GBSA.",
        "Reproducing older xtb results.",
        "tblite has no GBSA parameters for some solvents (ethanol, ...) and falls back to ALPB with a note.",
        "Like ALPB.",
        "Not applicable.",
        "Yes (fewer solvents than ALPB).",
        (R_GB, R_ALPB),
        aliases=("gb", "generalized born", "--gbsa"),
        see_also=("alpb", "solvation"),
    ),
    Entry(
        "cosmo", "COSMO", _FAMILY_SOLV,
        "Conductor-like screening model (MOPAC): surface charges for a perfect conductor, scaled to the solvent's ε.",
        "COSMO first finds the surface charges that would screen the solute completely in a conductor, then scales "
        "them down for a finite dielectric constant. C-PCM uses the same idea. QuantaBench passes MOPAC only the "
        "dielectric constant (`EPS=`), so no solvent-specific parameters are involved.",
        "Solvated PM7 and PM6 energies; the COSMO area and volume descriptors (MOPAC computes them in the gas phase "
        "too).",
        "MOPAC's COSMO is not smooth: energies and gradients jump as surface segments switch, so optimizations in "
        "COSMO can stop short and COSMO frequencies are approximate (CF3COO⁻ in water: a −28 cm⁻¹ spurious mode, "
        "docs/ENGINES.md).",
        "Small compared with the NDDO calculation.",
        "Not applicable.",
        "Yes: any solvent, from its dielectric constant.",
        (R_COSMO,),
        aliases=("conductor-like screening model",),
        see_also=("pcm", "pm7", "solvation"),
    ),
    Entry(
        "pcm", "PCM (IEF-PCM, C-PCM)", _FAMILY_SOLV,
        "Polarizable continuum model: Psi4's IEF-PCM, Spartan's C-PCM and tblite's CPCM; the DFT-level continuum.",
        "PCM puts apparent charges on the surface of a cavity made of atomic spheres so that they reproduce the "
        "dielectric's reaction field. IEF-PCM (Psi4, through PCMSolver) solves the dielectric problem; C-PCM "
        "(Spartan) and COSMO treat the solvent as a conductor and scale the result, which changes little for water "
        "(ε about 78). tblite's CPCM solves it with the ddX library, from the dielectric constant alone.",
        "DFT single points in water: dipoles, orbital energies, ESP and polarizabilities in the reaction field. "
        "tblite's CPCM is a cheap stand-in for the solvation part of a DFT/PCM conformer ranking: with GFN1-xTB "
        "charges its solvation energy differences between conformers of PFAS ions followed IEF-PCM's to 3.9 kJ/mol "
        "MAE, with GFN2-xTB to 9.7 (docs/VALIDATION.md, RANK-1).",
        "Electrostatics only in QuantaBench's setups (no cavity or dispersion terms, unlike [[smd|SMD]]), so PCM "
        "energies are not solvation free energies. Psi4 has no analytic PCM gradients: QM optimizations and "
        "frequencies run in the gas phase, and polarizabilities use finite fields. PCMSolver has no DMF, n-hexane, "
        "diethyl ether, octanol or ethyl acetate (gas phase with a note).",
        "In QuantaBench's Psi4 runs, PCM water took about 1.7x the gas-phase time for PFHpA at ωB97X-D/def2-SVP "
        "(857 vs 507 s), the extra time being the finite-field polarizability (docs/VALIDATION.md).",
        "Not applicable.",
        "Yes: the solvents PCMSolver knows (Psi4); any ε (tblite CPCM).",
        (R_PCM_REVIEW, R_CPCM, R_DDX),
        aliases=("cpcm", "c-pcm", "iefpcm", "ief-pcm", "ddcosmo", "pcm water"),
        see_also=("cosmo", "smd", "solvation"),
    ),
    Entry(
        "smd", "SMD", _FAMILY_SOLV,
        "Truhlar's solvation model: PCM-type electrostatics plus fitted surface terms, for solvation free energies.",
        "SMD adds to an IEF-PCM-type electrostatic term a CDS term (cavitation, dispersion, solvent structure) from "
        "atomic surface tensions fitted to experimental solvation free energies in many solvents. Its authors report "
        "mean unsigned errors of 0.6-1.0 kcal/mol for neutral solutes and about 4 kcal/mol for ions.",
        "Solvation free energies and partition coefficients at DFT level, in programs that have it.",
        "QuantaBench's Psi4 stage runs IEF-PCM, not SMD: keep that in mind when comparing with SMD numbers from "
        "other programs.",
        "Like PCM.",
        "Not applicable.",
        "Yes (not in QuantaBench).",
        (R_SMD,),
        aliases=("smd solvation",),
        see_also=("pcm", "solvation"),
    ),
    # ------------------------------------------------------------------ optimizers
    Entry(
        "lbfgs", "L-BFGS (ASE)", _FAMILY_OPT,
        "Quasi-Newton optimizer that estimates the curvature from recent steps; the default, reliable for molecules.",
        "Limited-memory BFGS keeps the last steps and gradient changes (100 in ASE) to estimate the curvature and "
        "takes Newton-like steps, at most 0.2 Å per atom. It stops when the largest atomic force is below Force "
        "convergence (0.03 eV/Å by default).",
        "Every engine; the batched GPU refinement has a batched version of it.",
        "Each step is one engine call: with external programs (xtb, MOPAC) their own optimizer is several times "
        "faster.",
        "One energy and gradient per step; tens of steps for a flexible molecule.",
        "Not applicable.",
        "Not applicable.",
        (R_LBFGS,),
        aliases=("l-bfgs", "bfgs"),
        see_also=("fire", "native-optimizer"),
    ),
    Entry(
        "fire", "FIRE (ASE)", _FAMILY_OPT,
        "Fast inertial relaxation engine: damped dynamics toward the minimum; tolerant of rough surfaces, often slower.",
        "FIRE moves the atoms like a damped molecular-dynamics run, speeding up while the force points downhill and "
        "stopping when it turns uphill. It needs no curvature estimate.",
        "Rough or noisy energy surfaces (COSMO, strained starting geometries) where L-BFGS steps go wrong.",
        "Often takes more steps than L-BFGS on well-behaved molecules.",
        "One energy and gradient per step.",
        "Not applicable.",
        "Not applicable.",
        (R_FIRE,),
        aliases=("fire optimizer",),
        see_also=("lbfgs",),
    ),
    Entry(
        "native-optimizer", "Engine's own optimizer", _FAMILY_OPT,
        "The program's own optimizer in one run (xtb, MOPAC): several times faster than ASE stepping it; else L-BFGS.",
        "xtb optimizes in approximate normal coordinates (ANCopt) and MOPAC with its own optimizer, each in one "
        "program run instead of one run per step. QuantaBench converts Force convergence into the program's "
        "gradient-norm criterion so that the result is at least as tight. Engines without one (dxtb, tblite, "
        "AIMNet2, MMFF94s) use L-BFGS.",
        "Refinements with `xtb-gfn2`, `xtb-gfn1`, `gfnff`, `mopac-pm7` and `mopac-pm6`.",
        "xtb exits successfully even when it runs out of cycles; QuantaBench reads its log and checks the final "
        "forces.",
        "xtb: 4-10x faster than ASE stepping xtb; MOPAC: 3-15x (docs/ENGINES.md).",
        "Not applicable.",
        "Not applicable.",
        (),
        aliases=("native", "ancopt", "engine optimizer"),
        see_also=("lbfgs",),
    ),
    # ------------------------------------------------------------------ Spartan
    Entry(
        "spartan-b3lyp", "Spartan B3LYP/6-311+G** (C-PCM water)", _FAMILY_SPARTAN,
        "Spartan's water numbers: B3LYP without dispersion, 6-311+G**, C-PCM water, single points at MMFF geometries.",
        "In the Spartan '24 results QuantaBench is validated against (docs/VALIDATION.md), the charged molecules "
        "have B3LYP/6-311+G** single points in C-PCM water (dielectric 78.3) at MMFF geometries. They give the "
        "water dipoles, HOMO/LUMO and ESP values that QuantaBench is compared with. The functional has no dispersion "
        "correction ([[b3lyp]]).",
        "Knowing what you compare with. QuantaBench's closest setting is the QM stage with `b3lyp/6-311+g**` in water "
        "(Psi4 uses IEF-PCM, not C-PCM).",
        "QuantaBench does not try to match these numbers exactly: its levels include dispersion. GFN2/ALPB reproduces "
        "these dipoles to 0.52 D MAE at Spartan's geometries (61 ions), but QuantaBench's own lowest conformers "
        "differ by 4.35 D MAE, 88 % of it from the conformer choice (VAL-3).",
        "Not applicable (Spartan's results).",
        "No.",
        "C-PCM water (ε 78.3).",
        (R_B3LYP, R_CPCM),
        aliases=("spartan b3lyp", "spartan water", "spartan dft water"),
        see_also=("b3lyp", "pcm", "6-311+g**", "dispersion"),
    ),
    Entry(
        "spartan-wb97x-d", "Spartan ωB97X-D/6-31G*", _FAMILY_SPARTAN,
        "Spartan's gas-phase DFT level for the Essential 76 PFAS; QuantaBench's Psi4 stage reproduces it at the same "
        "geometry.",
        "Spartan's ωB97X-D/6-31G* gas-phase single points on its conformers, with energies and dipoles per "
        "conformer. Its geometries come from DLFFG1, a neural-network estimate of ωB97X-D/6-31G* geometries, and its "
        "ranking from DLXD2XV, a neural-network estimate of ωB97X-V/6-311+G(2df,2p). At Spartan's geometry "
        "QuantaBench's Psi4 stage gives the same energies to 4e-4 Eh and dipoles to 0.001 D (VAL-2).",
        "Like-for-like checks: the QM stage with `wb97x-d/6-31g*` in the gas phase.",
        "For TFA and PFPrA Spartan's documents hold one anti acid conformer each, 17-19 kJ/mol above the syn one at "
        "GFN2 (VAL-3).",
        "Not applicable (Spartan's results).",
        "Yes: ωB97X-D's own term.",
        "Gas phase.",
        (R_WB97XD,),
        aliases=("spartan wb97x-d", "dlxd2xv", "spartan est. dft"),
        see_also=("wb97x-d", "6-31g*"),
    ),
    Entry(
        "spartan-pm6", "Spartan PM6", _FAMILY_SPARTAN,
        "Spartan's gas-phase semi-empirical level: plain PM6, without dispersion or hydrogen-bond corrections.",
        "Most neutral molecules have PM6 gas-phase optimizations with Spartan's QSAR properties (dipole, HOMO/LUMO, "
        "areas, volumes). Spartan's keyword is plain PM6.",
        "Knowing what you compare with in the gas phase.",
        "QuantaBench's `mopac-pm6` is [[pm6-d3h4x|PM6-D3H4X]], so its geometries and energies differ from Spartan's "
        "PM6 where dispersion or hydrogen bonds matter. QuantaBench measured GFN2's LUMO about 5.5 eV below PM6's "
        "and its HOMO 1.1 eV below (gap 4.4 eV smaller, r 0.80-0.88), and gas dipoles 1.41 D MAE from PM6's "
        "(r 0.65) (docs/dev/BACKLOG.md, VAL-3).",
        "Not applicable (Spartan's results).",
        "No.",
        "Gas phase.",
        (R_PM6,),
        aliases=("spartan pm6", "pm6 (spartan)"),
        see_also=("pm6-d3h4x", "homo-lumo"),
    ),
    # ------------------------------------------------------------------ concepts
    Entry(
        "dispersion", "Dispersion corrections (D3, D3(BJ), D4, -D)", _FAMILY_CONCEPTS,
        "Adds the London attraction most density functionals miss: it matters for folded chains, stacking, conformers.",
        "London dispersion is the attraction between atoms that are close but not bonded, from correlated "
        "fluctuations of their electrons; it falls off as 1/R⁶. Common functionals (B3LYP, PBE0) describe it "
        "poorly, so a correction adds damped −C6/R⁶ and −C8/R⁸ terms over atom pairs. D3 makes the C6 coefficients "
        "depend on each atom's coordination number; '(BJ)' is Becke-Johnson damping, which keeps a finite attraction "
        "at short distance instead of switching it off (zero damping, D3(0)). D4 makes them depend on atomic charges "
        "too. The '-D' of ωB97X-D is Chai and Head-Gordon's own older pairwise term; VV10 (in ωB97M-V) is a "
        "nonlocal functional for the same effect.",
        "In QuantaBench: AIMNet2 (D3(BJ)), GFN2-xTB and GFN-FF (D4), GFN1-xTB (D3(BJ)), PM6-D3H4X (D3), PM7 (built "
        "in), ωB97X-D (-D), B3LYP-D3(BJ) and PBE0-D3(BJ). None in plain B3LYP, HF or r2SCAN as listed; M06-2X only "
        "at medium range.",
        "A functional without one, like plain B3LYP in Spartan's runs, lacks the attraction between chain segments "
        "that touch, so it tends to favour extended conformers over folded ones ([[b3lyp]]). For PFAS ions in water, "
        "GFN2/ALPB (D4) and Spartan's B3LYP/C-PCM disagree on folded vs extended conformers by tens of kJ/mol "
        "(VAL-3). A DFT test found dispersion only a small part of that (D3(BJ): about 10 kJ/mol): ωB97X-D and "
        "B3LYP-D3(BJ) also put Spartan's pick below GFN2's in most of those ions, and GFN2/ALPB is the outlier.",
        "Negligible: milliseconds even for large molecules.",
        "This entry.",
        "Not applicable.",
        (R_DISP_REVIEW, R_D3, R_D4),
        aliases=("d3", "d3bj", "d3(bj)", "d4", "d2", "london dispersion", "-d", "vv10", "dispersion correction",
                 "becke-johnson"),
        see_also=("b3lyp", "b3lyp-d3bj", "wb97x-d", "gfn2-xtb"),
    ),
    Entry(
        "single-point", "Single point vs optimization", _FAMILY_CONCEPTS,
        "A single point computes the energy at a fixed geometry; an optimization first moves the atoms to a minimum.",
        "A single point is one SCF at the given coordinates: cheap, and the result belongs to that geometry, not to "
        "the method's own minimum. An optimization repeats energy-and-gradient steps until the forces vanish, so the "
        "geometry and the energy are consistent at one level. 'B3LYP/6-311+G**//MMFF' means a B3LYP single point at "
        "an MMFF geometry; Spartan's water numbers are of that kind.",
        "Single points: properties at a trusted geometry, or re-ranking many conformers cheaply. Optimizations: "
        "final structures, and frequencies (which need a minimum at the same level).",
        "Ranking conformers by single points at another method's geometries mixes two levels: a conformer that would "
        "relax a lot at the higher level is ranked too high. In Psi4 with PCM, optimizations run in the gas phase "
        "(no analytic PCM gradients).",
        "An optimization costs one energy and gradient per step, often tens of steps for a flexible molecule.",
        "Not applicable.",
        "Not applicable.",
        (),
        aliases=("single point", "optimization", "optimize", "energy", "sp", "opt", "//"),
        see_also=("crest-gfn2//gfnff", "spartan-b3lyp"),
    ),
    Entry(
        "boltzmann", "Boltzmann weighting", _FAMILY_CONCEPTS,
        "Weights each conformer by exp(−ΔE/RT): at 298 K a conformer 1.36 kcal/mol up gets a tenth of the weight.",
        "QuantaBench gives each kept conformer the weight w = exp(−ΔE/RT) / Σ exp(−ΔE/RT), with ΔE its refinement "
        "energy above the lowest conformer and T the Temperature setting (298.15 K by default). The "
        "Boltzmann-averaged descriptors (`dipole_debye_boltz` and the other `*_boltz` columns) use these weights. RT "
        "is 0.593 kcal/mol (2.48 kJ/mol) at 298 K, so a conformer 4.1 kcal/mol up gets 1/1000 of the lowest's "
        "weight.",
        "Descriptors of flexible molecules, whose volume, area and dipole change between conformers.",
        "The weights use electronic energies at the refinement level (in a solvent, including the ALPB or COSMO "
        "solvation energy), not free energies with vibrational entropy. An error of 1-2 kcal/mol in a relative "
        "energy changes a weight by a factor of 5-30, so an average is only as good as the ranking. Conformers "
        "outside the energy window are dropped before weighting.",
        "Free.",
        "Not applicable.",
        "Not applicable.",
        (),
        aliases=("boltzmann", "boltzmann average", "weights", "conformer weights"),
        see_also=("rdkit-etkdg", "crest"),
    ),
    Entry(
        "homo-lumo", "HOMO and LUMO energies", _FAMILY_CONCEPTS,
        "xTB and PM6/PM7 orbital energies are trend descriptors: use them to rank molecules, not as absolute energies.",
        "The HOMO and LUMO are the highest occupied and the lowest unoccupied molecular orbitals. In Hartree-Fock, "
        "−ε_HOMO approximates the ionization energy (Koopmans' theorem); in DFT, and even more in minimal-basis "
        "tight binding, orbital energies depend strongly on the method. The GFN methods were fitted to geometries, "
        "frequencies and noncovalent energies, not to orbital energies.",
        "QSAR descriptors within one method and one phase; comparing families of molecules.",
        "Methods are offset from each other by electronvolts. In QuantaBench's 300-molecule validation GFN2's LUMO "
        "sits about 5.5 eV below PM6's and its HOMO 1.1 eV below, so its gap is 4.4 eV smaller, though the two "
        "correlate (r 0.80-0.88; docs/dev/BACKLOG.md, VAL-3). Use one method for a whole data set. In the gas phase "
        "the HOMO of a multiply charged anion can lie above zero (unbound).",
        "Free with any SCF method (xTB, NDDO, DFT).",
        "Not applicable.",
        "The solvent shifts orbital energies: compare within one phase.",
        (R_KOOPMANS, R_GFN2),
        aliases=("homo", "lumo", "gap", "homo-lumo gap", "orbital energies"),
        see_also=("gfn2-xtb", "spartan-pm6", "hf"),
    ),
    Entry(
        "gcp", "Basis-set superposition error and gCP", _FAMILY_CONCEPTS,
        "In a small basis, touching atoms borrow each other's functions and bind too strongly; gCP corrects that.",
        "In a finite basis set the electrons of one atom can use the functions centred on its neighbours. Two parts "
        "of a molecule that touch are then bound a little too strongly: the basis-set superposition error (BSSE). It "
        "is largest in small basis sets and makes folded, compact conformers too low. The counterpoise correction "
        "(Boys and Bernardi) removes it for complexes; gCP (Kruse and Grimme) is a cheap geometric estimate that also "
        "works within one molecule, and the [[r2scan-3c|-3c composites]] include it.",
        "Knowing why def2-SVP and 6-31G* lean toward compact conformers, and why r2SCAN-3c needs no larger basis.",
        "QuantaBench measured the small-basis lean against r2SCAN-3c: ωB97X-D/def2-SVP in PCM favours compact "
        "conformers by +15.0 ± 2.4 kJ/mol per Å of radius of gyration for PFAS ions and +7.8 ± 1.0 for "
        "non-fluorinated ions; in the gas phase +3.7 ± 1.9 against def2-TZVP for neutrals, 1-2 kJ/mol across their "
        "pools (docs/VALIDATION.md, RANK-1b). For anions part of it is the missing diffuse functions. The QM stage "
        "has no gCP option: use a diffuse or triple-zeta basis instead.",
        "gCP itself costs nothing; a larger basis does (def2-SVPD about 3x, def2-TZVP about 7x def2-SVP).",
        "Not applicable.",
        "Not applicable.",
        (R_GCP, R_COUNTERPOISE),
        aliases=("bsse", "basis set superposition error", "counterpoise", "counterpoise correction"),
        see_also=("r2scan-3c", "def2-svp", "def2-svpd", "basis-sets"),
    ),
    Entry(
        "conformer-ranking", "Ranking conformers (which level picks the lowest)", _FAMILY_CONCEPTS,
        "Which method decides the lowest conformer: QuantaBench's measured choices by phase and molecule class.",
        "After the refinement, the rank step (`--rank`, Settings: the Conformer ranking page) gives every refined "
        "conformer a single point at the rank level, at its refined geometry. Those energies set the lowest conformer, "
        "the rank window (30 kJ/mol by default) and the Boltzmann weights, and so every descriptor taken from the "
        "lowest conformer or the ensemble (dipole, `xtb_*`, frequencies, the QM stage's conformers). Before the rank "
        "step only RMSD duplicates are pruned, never by refinement energy. A method can give good geometries and "
        "still rank conformers poorly, because conformer energy differences are a few kJ/mol and its errors can be "
        "larger. QuantaBench measured cheap rankers against DFT references on conformer pools of PFAS and "
        "non-fluorinated molecules (RANK-1, RANK-1b), and the rank step reproduces those picks (RANK-2).",
        "Choosing the rank level. `auto` (the default) ranks by phase: in water [[g-xtb|g-xTB]] gas + a GFN1-xTB "
        "CPCM ΔG_solv (`gxtb+cpcm`; 38 of 40 neutrals and 12 of 13 PFAS ions within 4 kJ/mol of r2SCAN-3c/PCM), "
        "falling back to AIMNet2 + that term and then the refinement; in the gas phase AIMNet2 (39 of 40 "
        "non-fluorinated neutrals, 18 of 20 PFAS), else g-xTB (40 of 40, 18 of 20), else the refinement. `refine` "
        "keeps the refinement's energies (fine for pure hydrocarbons: GFN2-xTB 12 of 12); `qm` ranks every conformer "
        "at the QM stage's level (docs/VALIDATION.md, RANK-1b; docs/CHOOSING_METHODS.md).",
        "GFN2-xTB ranks polar molecules poorly in water (16 of 28 polar organics, 14 of 20 neutral PFAS with ALPB) "
        "and PFAS ions no better than a random pick in every solvent model (0 of 13 against r2SCAN-3c, 4 of 17 "
        "against ωB97X-D/def2-SVP); its 6 kcal/mol window dropped DFT's lowest in 6 of 17 ions (RANK-1), so do not "
        "rank polar molecules with `refine`. No cheap ranker did well on non-fluorinated ions in water (5 of 8, flat "
        "pools). Where descriptors must be at DFT quality, a QM stage on the rank step's lowest conformers checks it "
        "(r2SCAN-3c or a diffuse basis in PCM, not def2-SVP; RANK-1b).",
        "A single point per conformer at the rank level: in water 0.3-1.3 s per conformer on one thread for "
        "g-xTB + CPCM (most of it the CPCM SCF), about 0.1 % of a DFT single point; in the gas phase 0.1 s for g-xTB "
        "and 0.01 s for AIMNet2 (after a few seconds to load its model once per process); minutes per conformer for "
        "DFT (RANK-2).",
        "Not applicable.",
        "In water the rank level needs a solvation term: g-xTB in the gas phase alone picked within 4 kJ/mol in 11 of "
        "17 PFAS ions, with the GFN1-xTB CPCM term 13 (RANK-1).",
        (),
        aliases=("conformer ranking", "ranking", "rank step", "rank-2", "lowest conformer", "rank method",
                 "rank_method", "rank window"),
        see_also=("gfn2-xtb", "aimnet2", "g-xtb", "pcm", "boltzmann"),
    ),
    Entry(
        "protonation", "Protonation state at pH", _FAMILY_CONCEPTS,
        "Acids lose and bases gain a proton at the run's pH (built-in rules); the charge then sets everything after.",
        "Before the conformer search QuantaBench neutralizes each structure, then applies a small table of acid and "
        "base sites with typical pKa values at the pH setting (7 by default): carboxylic, sulfonic and phosphonic "
        "acids lose a proton, aliphatic amines and amidines gain one, and perfluoroalkyl sulfonamides (pKa about 6.5) "
        "lose their N-H proton. Dimorphite-DL can replace the table. `--no-protonate` keeps the charges as drawn.",
        "Water runs of acids and bases, where the ion is the species present.",
        "One state per molecule, the one the rule's pKa makes dominant; near a pKa both forms matter, and sulfonamide "
        "pKa values vary with substitution. With `--no-protonate` a record drawn with the proton in the wrong place "
        "keeps it: one ion of the DFT conformer test was drawn with the proton on its sulfonamide N, 183 kJ/mol "
        "(ωB97X-D/PCM) above the amine-protonated cation (docs/VALIDATION.md).",
        "Free.",
        "Not applicable.",
        "The charge state is chosen for water at that pH; a gas-phase run of the same ion stays charged.",
        (),
        aliases=("ph", "pka", "charge state", "tautomer", "protonation state"),
        see_also=("gas-phase", "solvation"),
    ),
    Entry(
        "auto", "Auto (best available)", _FAMILY_CONCEPTS,
        "Takes the best installed engine for the molecule's elements: AIMNet2, then GFN2-xTB, then GFN1-xTB, then MMFF94s.",
        "QuantaBench tries AIMNet2, GFN2-xTB (dxtb, then tblite, then the xtb program) and GFN1-xTB in that order and "
        "takes the first one that is installed and has parameters for every element. With a solvent it tries the "
        "engines that can model it first. When it passes over an engine for a molecule, the notes say why.",
        "Mixed batches where some molecules have elements AIMNet2 lacks (Sn, Hg, metals).",
        "Molecules of one batch can end up at different levels: check the method column of summary.csv before "
        "comparing energies or dipoles. With a solvent, AIMNet2 and dxtb (gas phase only) move behind tblite and xtb.",
        "That of the engine it takes.",
        "That of the engine it takes (all candidates have one).",
        "Prefers an engine with a solvent model when a solvent is set.",
        (),
        aliases=("auto (best available)", "best available"),
        see_also=("aimnet2", "gfn2-xtb", "gfn1-xtb", "mmff94s"),
    ),
    Entry(
        "density-girolami", "Girolami liquid density", _FAMILY_EST,
        "A graph-only liquid density from relative atom volumes, for every molecule: `density_girolami_g_cm3`.",
        "Girolami's rule sums relative atomic volumes (H 1, Li-F 2, Na-Cl 4, K-Br 5, Rb-I 7.5, Cs-Bi 9) and divides "
        "the molar mass by five times the sum, giving g/cm³. The density is raised 10 % for each OH, COOH, primary or "
        "secondary amine, amide, sulfoxide and ring, 20 % for a sulfone and 7.5 % per ring of a fused system, at most "
        "30 % in all. QuantaBench writes the density and the molar volume M/ρ (`molar_volume_girolami_cm3`, cm³/mol) "
        "of every neutral molecule in the descriptors step.",
        "A density where none is measured, as a cross-check of [[density-packing|the packing density]], and the "
        "molar volume V_m that a parachor model σ = (P/V_m)⁴ needs.",
        "QuantaBench measured 5.3 % mean error on 237 PubChem test liquids, 6.4 % on fluorinated liquids (≥ 3 F) and "
        "13 % on phosphorus compounds (docs/VALIDATION.md, DESC-2). Fluorine counts as any second-row atom. Not "
        "defined for ions (no value, with a note).",
        "None: it reads the molecular graph.",
        "Not applicable.",
        "None: a pure-liquid property at room temperature.",
        (R_GIROLAMI,),
        aliases=("girolami", "density_girolami_g_cm3", "molar_volume_girolami_cm3", "girolami density"),
        see_also=("density-packing", "parachor-sugden"),
    ),
    Entry(
        "density-packing", "Packing-coefficient liquid density", _FAMILY_EST,
        "Liquid density from the computed van der Waals volume and a fitted packing coefficient: `density_packing_g_cm3`.",
        "The liquid's molar volume is the molecules' van der Waals volume divided by how densely they pack: V_m = "
        "N_A V_vdW / k. QuantaBench computes V_vdW with Bondi's radii at the best conformer and takes k (about 0.52-"
        "0.64) from a linear model in the inverse number of heavy atoms, the fractions of F, Cl, Br, I, Si, S, P, N "
        "and O, the H-bond donors and the ring and aromatic atoms, fitted on 831 PubChem/HSDB liquid densities near "
        "25 °C (DESC-2). It writes the density and `molar_volume_packing_cm3` (cm³/mol) for every neutral molecule.",
        "The molar volume of a parachor model when no density is measured, PFAS included; the best of the cheap "
        "estimates.",
        "QuantaBench measured 2.7 % mean error on 237 test liquids it was not fitted to, 3.2 % on 26 fluorinated "
        "liquids (3.8 % on the 11 it was not fitted to) and 3.0 % above 1.6 g/cm³ (docs/VALIDATION.md, DESC-2). Fitted for H, C, N, O, F, Si, P, S, Cl, Br, "
        "I only (other elements get no value), on neutral liquids at 1 atm; with non-default radii it computes its "
        "own Bondi volume.",
        "One grid volume of the best conformer: milliseconds.",
        "Not applicable.",
        "None: a pure-liquid property at room temperature.",
        (R_BONDI, R_GCVOL),
        aliases=("packing density", "density_packing_g_cm3", "molar_volume_packing_cm3", "packing coefficient"),
        see_also=("density-girolami", "parachor-sugden"),
    ),
    Entry(
        "polarizability-bosque", "Bosque-Sales additive polarizability", _FAMILY_EST,
        "Static polarizability from the composition alone: `polarizability_bosque_A3`.",
        "Bosque and Sales fitted average atomic polarizabilities to the experimental polarizabilities of 340 "
        "solvents: α (Å³) = 0.32 + 1.51 n_C + 0.17 n_H + 0.57 n_O + 1.05 n_N + 2.99 n_S + 2.48 n_P + 0.22 n_F + "
        "2.16 n_Cl + 3.29 n_Br + 5.45 n_I. It ignores structure and charge.",
        "An absolute polarizability that scales with size, for dispersion terms of a QSPR (α/V, the quantum "
        "parachor); PFAS included.",
        "QuantaBench measured 2.8 % mean error against Lorentz-Lorenz polarizabilities of 696 PubChem liquids, 3.5 % "
        "against ωB97X-D/def2-SVPD (3.7 % for PFAS) and 4.8 % against 27 experimental values (water 18 % off) "
        "(docs/VALIDATION.md, DESC-2). No value for other elements (Si, B, metals) or isomers' differences.",
        "None.",
        "Not applicable.",
        "None: a gas-phase molecular property.",
        (R_BOSQUE, R_MILLER),
        aliases=("bosque", "bosque-sales", "polarizability_bosque_A3", "additive polarizability"),
        see_also=("polarizability-crippen", "polarizability-d4", "polarizability-xtb"),
    ),
    Entry(
        "polarizability-crippen", "Crippen molar-refractivity polarizability", _FAMILY_EST,
        "Polarizability from Wildman-Crippen atom-type refractivities: `polarizability_crippen_A3`.",
        "RDKit's Wildman-Crippen molar refractivity MR is a sum of atom-type contributions fitted to measured molar "
        "refractivities. Lorentz-Lorenz turns it into a polarizability: α = 3 MR / (4π N_A), 0.3964 Å³ per cm³/mol. "
        "It is an optical (589 nm) polarizability, a few percent above the static one.",
        "Like [[polarizability-bosque|Bosque-Sales]], with atom types (hybridization, neighbours) instead of plain "
        "element counts.",
        "QuantaBench measured 1.7 % against Lorentz-Lorenz values of PubChem liquids (it was fitted to that kind of "
        "data), 4.3 % against ωB97X-D/def2-SVPD (3.5 % for PFAS) and 3.8 % against 27 experimental values "
        "(docs/VALIDATION.md, DESC-2). Only H, C, N, O, F, P, S, Cl, Br and I have their own types: other elements "
        "get no value.",
        "None.",
        "Not applicable.",
        "None: a gas-phase molecular property.",
        (R_CRIPPEN,),
        aliases=("crippen", "wildman-crippen", "molar refractivity", "polarizability_crippen_A3", "crippen mr"),
        see_also=("polarizability-bosque", "polarizability-d4"),
    ),
    Entry(
        "polarizability-d4", "D4-model polarizability", _FAMILY_EST,
        "The D4 dispersion model's static polarizability with EEQ charges at the best conformer: `polarizability_d4_A3`.",
        "D4 (the [[dispersion]] correction) weights TD-DFT reference polarizabilities of each element by the atom's "
        "coordination number and its electronegativity-equalization (EEQ) charge. The sum over atoms is a molecular "
        "static polarizability that responds to the environment and the charge, with no SCF. QuantaBench computes it "
        "with tad-dftd4 (or the copy inside the dxtb fork) when PyTorch is installed; without them the column is "
        "absent.",
        "A geometry- and charge-aware polarizability for every element up to Rn, silicon and metals included.",
        "QuantaBench measured 1.02 x DFT (ωB97X-D/def2-SVPD) for non-fluorinated molecules but 1.11 x DFT for PFAS, "
        "and 0.96 x Lorentz-Lorenz (r 0.991) for PubChem liquids (docs/VALIDATION.md, DESC-2): rescale it before "
        "mixing fluorinated and other molecules. It is not the xTB response ([[polarizability-xtb]]).",
        "Milliseconds after PyTorch is loaded.",
        "Not applicable.",
        "None: the gas-phase EEQ charges.",
        (R_D4,),
        aliases=("polarizability_d4_A3", "d4 polarizability", "d4 alpha", "dftd4 polarizability"),
        see_also=("dispersion", "polarizability-bosque", "polarizability-xtb"),
    ),
    Entry(
        "polarizability-xtb", "xTB response polarizability (GFN1, GFN2)", _FAMILY_EST,
        "The SCF response of GFN1-/GFN2-xTB to a field: `xtb_polarizability_iso_A3`, `tblite_gfn2_polarizability_iso_A3`.",
        "The static polarizability of the [[gfn1-xtb|GFN1-xTB]] and [[gfn2-xtb|GFN2-xTB]] Hamiltonians: dxtb's "
        "autograd or tblite's finite field (six SCFs in ±field), with the run's solvent when it has one. Every run "
        "with tblite now writes both, whatever engine refined the conformers. The minimal valence basis cannot "
        "polarize much, so the values are well below experiment.",
        "Relative trends within one class, solvent shifts (water − gas), and recalibrated (one scale factor per "
        "class) when DFT is too expensive.",
        "QuantaBench measured GFN2-xTB at 0.66 x and GFN1-xTB at 0.60 x ωB97X-D/def2-SVPD, with ratios from 0.45 to "
        "1.4 (high for bromine and iodine), and 0.65 x / 0.56 x Lorentz-Lorenz (docs/VALIDATION.md, DESC-2). As the "
        "α of a parachor model it fails without a volume term (b02). Gas-phase multiply charged anions can have "
        "unphysical values (discarded with a note).",
        "Six or seven SCFs: milliseconds to a second.",
        "Not applicable.",
        "The run's solvent (ALPB by default) in the same SCF as the field.",
        (R_GFN2, R_GFN1),
        aliases=("xtb polarizability", "xtb_polarizability_iso_A3", "tblite_gfn2_polarizability_iso_A3",
                 "gfn2 polarizability", "gfn1 polarizability"),
        see_also=("gfn1-xtb", "gfn2-xtb", "polarizability-d4", "polarizability-bosque"),
    ),
    Entry(
        "parachor-sugden", "Sugden parachor", _FAMILY_EST,
        "Sugden's group parachor P, with σ ≈ (P ρ / M)⁴ for a pure liquid: `parachor_sugden`.",
        "The parachor P = M σ^¼ / (ρ_l − ρ_v) is nearly additive over atoms and bonds. QuantaBench sums Sugden's "
        "values as Quayle tabulated them (C 4.8, H 17.1, O 20.0, F 25.7, Cl 54.3, Br 68.0, ...; a double bond "
        "23.2, a six-ring 6.1, an ester group 60.0), in cm³ mol⁻¹ (mN/m)^¼. With a liquid density, measured or "
        "[[density-packing|estimated]], it predicts the surface tension with no fit.",
        "A surface-tension baseline for any liquid with C, H, N, O, F, Si, P, S, Cl, Br, I or B.",
        "QuantaBench measured on b02 (80 non-fluorinated liquids): 1.7 mN/m mean error with the measured density, "
        "3.1 with the packing density, 6.7 with Girolami's (docs/SURFACE_TENSION.md, docs/VALIDATION.md, DESC-2). "
        "Errors in P or ρ count four times in σ. Pure liquids only: not the surface tension of a surfactant "
        "solution.",
        "None: it reads the molecular graph.",
        "Not applicable.",
        "None: a pure-liquid property.",
        (R_SUGDEN, R_QUAYLE),
        aliases=("sugden", "parachor", "parachor_sugden"),
        see_also=("density-packing", "density-girolami"),
    ),
)

#: What each method or correction is for, one line each: (corrects, use for, avoid for). The model advisor
#: (``chem.advisor``) links its findings to these entries; numbers are QuantaBench's own (docs/VALIDATION.md).
DECISIONS: dict[str, tuple[str, str, str]] = {
    "dispersion": (
        "The London dispersion attraction most functionals and HF miss: without it, folded conformers, stacked rings "
        "and touching chain segments come out too high.",
        "Every DFT conformer energy: pick a level with -D3(BJ), -D4, -D or -V.",
        "Nothing: it costs milliseconds. The Spartan B3LYP and PM6 results QuantaBench is validated against "
        "have none, which is the cautionary case."),
    "b3lyp-d3bj": (
        "Plain B3LYP's missing dispersion, with D3 and Becke-Johnson damping.",
        "B3LYP comparisons with the dispersion fixed.",
        "Ranking anion conformers at def2-SVP: it shares that basis' lean toward compact conformers (RANK-1)."),
    "b3lyp": (
        "Nothing: no dispersion correction.",
        "Reproducing Spartan or older B3LYP numbers.",
        "New conformer energies: folded conformers come out too high."),
    "wb97x-d": (
        "Dispersion (its own -D term); long-range exact exchange treats anions and orbital energies better than "
        "global hybrids.",
        "DFT single points and properties; QuantaBench's default QM functional.",
        "Ranking anion conformers at def2-SVP: use def2-SVPD (RANK-1)."),
    "pbe0-d3bj": ("Dispersion, through D3(BJ).", "A second hybrid next to B3LYP or ωB97X-D.", ""),
    "r2scan": (
        "Medium-range dispersion only, from the functional.",
        "Larger molecules when hybrids are too slow, with -D4 added.",
        "Conformer energies without -D4."),
    "m06-2x": (
        "Medium-range dispersion, through its fit.",
        "Reaction energies and barriers.",
        "Long-range contacts without an added D3 term."),
    "hf": (
        "Nothing: no electron correlation, no dispersion.",
        "A baseline; Koopmans ionization energies.",
        "Conformer energies."),
    "r2scan-3c": (
        "Bundles what a small basis lacks: D4 dispersion and gCP for basis-set superposition error, in a matched "
        "triple-zeta basis.",
        "A DFT reference for conformer energies (RANK-1b's).",
        "The QM stage today: not selectable yet."),
    "gcp": (
        "Basis-set superposition error: folded conformers and complexes bound too strongly in a small basis.",
        "Small-basis DFT and HF; built into the -3c composites.",
        ""),
    "basis-sets": (
        "Diffuse functions: the loosely bound electrons of anions, which a compact basis squeezes. Polarization "
        "functions: bonds that bend and polarize.",
        "A diffuse basis (def2-SVPD, 6-31+G*) for anions, polarizabilities and ESP surfaces.",
        "A compact basis (def2-SVP, 6-31G*) for ranking anion conformers."),
    "def2-svp": (
        "",
        "Fast DFT of neutral molecules.",
        "Ranking anion conformers: +15 kJ/mol per Å of radius of gyration toward compact ones for PFAS ions "
        "(RANK-1b)."),
    "def2-svpd": (
        "The loosely bound electrons of anions (diffuse functions).",
        "Anions: conformer ranking, polarizabilities, ESP.",
        "Large neutral batches, where it adds about 3x the cost for little."),
    "def2-tzvp": (
        "Most of a double-zeta basis' superposition error.",
        "Final energies and properties of neutral molecules.",
        "Anions, unless a diffuse set is added."),
    "6-31g*": ("", "Like-for-like checks against Spartan's ωB97X-D/6-31G*.",
               "Anions; elements beyond Kr (Psi4 has it for H-Kr only)."),
    "6-31+g*": ("The loosely bound electrons of anions (diffuse s and p on heavy atoms).", "Anions at modest cost.",
                "Elements beyond Ca (Psi4 has it for H-Ca only)."),
    "6-311+g**": ("Anions (diffuse functions) and polarization on hydrogen.",
                  "Like-for-like checks against Spartan's B3LYP/6-311+G** water numbers.",
                  "Elements beyond Ca (Psi4 has it for H-Ca only)."),
    "aug-cc-pvdz": ("Diffuse functions on every atom, hydrogen included.", "Anions, polarizabilities, weak complexes.",
                    "Large molecules (nearly linearly dependent functions); K and Ca (not in Psi4's set)."),
    "solvation": (
        "The solvent's screening of charges and its solvation energy, which decide the conformers and dipoles of "
        "ions and polar molecules.",
        "Ions and polar molecules in water.",
        ""),
    "gas-phase": ("Nothing: the isolated molecule.", "Neutral molecules; gas-phase reference data.",
                  "Ions: their charges are unscreened."),
    "alpb": (
        "Electrostatic screening (generalized Born) plus non-polar surface terms for the cavity and dispersion with "
        "the solvent.",
        "xTB geometries and properties in water.",
        "Ranking polar conformers: its solvation differences missed IEF-PCM's by 20 kJ/mol for PFAS ions (RANK-1)."),
    "gbsa": ("Like ALPB: generalized Born screening plus a surface-area term.", "Reproducing older xtb results.",
             "New work: ALPB replaced it."),
    "pcm": (
        "Electrostatic screening by the solvent (the reaction field). As QuantaBench sets it up it has no cavity or "
        "dispersion terms, which favours extended chains of surfactant-like ions (RANK-1b).",
        "DFT single points in water; with GFN1-xTB charges (tblite's CPCM), the solvation term of a conformer "
        "ranking (RANK-1).",
        "Solvation free energies (use SMD in another program)."),
    "cosmo": ("Electrostatic screening, as a conductor scaled to the dielectric constant.",
              "Solvated PM7 and PM6-D3H4X energies; COSMO areas and volumes.",
              "Frequencies in solvent (MOPAC's COSMO is not smooth)."),
    "smd": (
        "Electrostatics plus fitted non-electrostatic terms (cavity, dispersion, solvent structure): solvation free "
        "energies.",
        "Solvation free energies, in programs that have it.",
        "QuantaBench today: not available."),
    "pm6-d3h4x": (
        "Plain PM6's dispersion (D3), hydrogen-bond (H4) and halogen-contact (X) errors.",
        "Noncovalent contacts at semi-empirical cost.",
        "Ranking PFAS anion conformers: it overbinds short O···F contacts like GFN2-xTB (RANK-1)."),
    "pm7": ("Dispersion and hydrogen bonds, with terms built into the method.",
            "Heats of formation; MOPAC's QSAR properties.",
            "Ranking the conformers of PFAS ions (RANK-1)."),
    "gfn2-xtb": (
        "GFN1's charge-only electrostatics: atomic dipoles and quadrupoles describe anisotropic contacts (hydrogen "
        "and halogen bonds); D4 dispersion is computed self-consistently.",
        "Geometries in the gas phase and in ALPB; ranking pure hydrocarbon conformers (12 of 12, RANK-1b).",
        "Ranking the conformers of ions and polar molecules, in any solvent model (RANK-1, RANK-1b)."),
    "gfn1-xtb": ("Halogen bonds, with an extra term.",
                 "The CPCM solvation term of a conformer ranking (RANK-1); the `xtb_*` properties.", ""),
    "g-xtb": ("", "Ranking conformers: the rank step, gas phase, and with a GFN1-xTB CPCM term in water (RANK-1, "
                  "RANK-1b).", "Geometries (not a refinement engine here); its own provisional solvation models."),
    "aimnet2": (
        "", "Neutral molecules in the gas phase: geometries and conformer ranking (RANK-1, RANK-1b).",
        "Open shells (a closed-shell model), elements outside its 14, solvent runs (no implicit solvent)."),
    "mmff94s": ("", "Pre-optimizing the conformer pool.", "The final conformer ranking; electronic properties."),
    "gfn-ff": ("", "Fast CREST searches of large or unusual molecules.", "Ranking conformers: a search level only."),
    "crest": ("", "Flexible molecules and ions in water; conformers ETKDG misses.",
              "Large batches of large flexible molecules at GFN2 (hours each)."),
    "rdkit-etkdg": ("", "Fast, reproducible conformer pools.",
                    "Ions in water with few conformers refined: the pool is ranked by gas-phase MMFF94s."),
    "density-girolami": (
        "The missing liquid density, from the graph alone.",
        "A quick density and molar volume where none is measured; a cross-check of the packing density.",
        "Final densities of fluorinated or phosphorus compounds (6-13 % off, DESC-2)."),
    "density-packing": (
        "The gap between the van der Waals volume and the liquid's molar volume (V_vdW is 50-66 % of V_m), with a "
        "packing coefficient fitted to measured densities.",
        "V_m of a parachor model with no measured density, PFAS included (2.7 % mean error, DESC-2).",
        "Ions, elements it was not fitted for, liquids far from 25 °C and 1 atm."),
    "polarizability-bosque": (
        "The low absolute polarizability of the minimal-basis xTB response.",
        "Absolute α for dispersion descriptors (α/V, the quantum parachor).",
        "Si, B, metals (no parameters); isomers (it sees only the composition)."),
    "polarizability-crippen": (
        "The low xTB response, with atom types fitted to measured refractivities.",
        "Absolute α where the atom types matter.",
        "Elements outside H C N O F P S Cl Br I."),
    "polarizability-d4": (
        "The low xTB response, with no SCF; responds to geometry and charge.",
        "Molecules with elements the additive tables lack (Si, metals).",
        "Mixing PFAS and other molecules without a rescale (11 % high for PFAS, DESC-2)."),
    "polarizability-xtb": (
        "",
        "Trends within a class and solvent shifts of α.",
        "Absolute polarizabilities: half of DFT's for many molecules (DESC-2)."),
    "parachor-sugden": (
        "",
        "A no-fit surface-tension baseline, with a measured or estimated density.",
        "Surfactant solutions (γ at the CMC) and ions."),
}


def _with_decisions(e: Entry) -> Entry:
    corrects, use_for, avoid_for = DECISIONS.get(e.key, ("", "", ""))
    return replace(e, corrects=corrects, use_for=use_for, avoid_for=avoid_for)


ENTRIES: dict[str, Entry] = {e.key: _with_decisions(e)
                             for e in sorted(_ENTRIES, key=lambda e: FAMILIES.index(e.family))}

#: engine key (``engines.all_engines()``, plus "auto") -> entry key
ENGINE_ENTRIES: dict[str, str] = {
    "auto": "auto", "aimnet2": "aimnet2", "gfn2-xtb": "gfn2-xtb", "tblite-gfn2": "gfn2-xtb", "xtb-gfn2": "gfn2-xtb",
    "gfn1-xtb": "gfn1-xtb", "tblite-gfn1": "gfn1-xtb", "xtb-gfn1": "gfn1-xtb", "mopac-pm7": "pm7",
    "mopac-pm6": "pm6-d3h4x", "mmff": "mmff94s", "gfnff": "gfn-ff",
}
#: BuildConfig.conformer_search -> entry key
CONFORMER_SEARCH_ENTRIES: dict[str, str] = {"rdkit": "rdkit-etkdg", "crest": "crest"}
#: BuildConfig.crest_method -> entry key
CREST_LEVEL_ENTRIES: dict[str, str] = {"gfn2": "crest-gfn2", "gfnff": "crest-gfnff", "gff": "crest-gfnff",
                                       "gfn2//gfnff": "crest-gfn2//gfnff", "gfn1": "crest-gfn1"}
#: BuildConfig.optimizer -> entry key
OPTIMIZER_ENTRIES: dict[str, str] = {"lbfgs": "lbfgs", "fire": "fire", "native": "native-optimizer"}
#: solvation model (``EngineInfo.solvation``, Psi4's "pcm"; "" = gas phase) -> entry key
SOLVATION_MODEL_ENTRIES: dict[str, str] = {"": "gas-phase", "alpb": "alpb", "gbsa": "gbsa", "cpcm": "pcm",
                                           "cosmo": "cosmo", "pcm": "pcm"}


# ============================================================================== coverage: which molecules a method takes
SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr "
           "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
           "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()


def _z(*spans) -> frozenset[int]:
    """Atomic numbers from single numbers and (first, last) pairs."""
    out: set[int] = set()
    for s in spans:
        out.update(range(s[0], s[1] + 1) if isinstance(s, tuple) else (s,))
    return frozenset(out)


def element_text(numbers) -> str:
    """'H-Rn', 'H-La, Lu-Bi', 'H, B, C, ...': runs of six or more consecutive elements as ranges."""
    zs = sorted({int(z) for z in numbers if 0 < int(z) < len(SYMBOLS)})
    parts, i = [], 0
    while i < len(zs):
        j = i
        while j + 1 < len(zs) and zs[j + 1] == zs[j] + 1:
            j += 1
        if j - i >= 5:
            parts.append(f"{SYMBOLS[zs[i]]}-{SYMBOLS[zs[j]]}")
        else:
            parts += [SYMBOLS[z] for z in zs[i:j + 1]]
        i = j + 1
    return ", ".join(parts)


Z_H_RN = _z((1, 86))
#: AIMNet2's 14 elements (engines.AIMNET2_ELEMENTS; a test checks they agree)
Z_AIMNET2 = _z(1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53)
#: MMFF94's organic elements and the free ions it has types for (Halgren 1996)
Z_MMFF94 = _z(1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
Z_MMFF94_IONS = _z(3, 11, 12, 19, 20, 26, 29, 30)
#: PM6 and PM7 in MOPAC: H-La and Lu-Bi (Ce-Yb only as Sparkles; eng_mopac.ELEMENTS)
Z_NDDO = _z((1, 57), (71, 83))
#: the def2 basis sets as Psi4 ships them: H-Rn without Ce-Lu
Z_DEF2 = _z((1, 57), (72, 86))


@dataclass(frozen=True)
class Coverage:
    """Which molecules a method (or basis set, or CREST level) can take: elements, charge and spin. The model
    advisor (``chem.advisor``) never recommends a method for a molecule outside its coverage."""

    key: str                       # the method's guide entry (COVERAGE is keyed by it)
    label: str
    elements: frozenset[int]
    open_shell: str = "yes"        # "yes" (unrestricted / spin-polarized), "no" (closed shells only), "ignored"
                                   # (a force field: no electrons, the spin is not modelled)
    charged: bool = True
    solvation: str = ""            # what it does in a solvent run
    available: str = ""            # "" = QuantaBench runs it; else where it can run instead and why not here
    note: str = ""
    source: str = ""               # where the element list comes from
    verified: str = ""             # how QuantaBench checked it
    extra_elements: frozenset[int] = field(default_factory=frozenset)   # e.g. MMFF94's free ions

    def missing(self, numbers) -> list[int]:
        """Atomic numbers of ``numbers`` it has no parameters for."""
        return sorted({int(z) for z in numbers} - self.elements - self.extra_elements)

    def handles(self, numbers, charge: int = 0, multiplicity: int = 1) -> tuple[bool, str]:
        """(ok, why not) for a molecule of these elements, total charge and spin multiplicity."""
        missing = self.missing(numbers)
        if missing:
            return False, f"{self.label} has no parameters for {', '.join(SYMBOLS[z] for z in missing)}"
        if charge and not self.charged:
            return False, f"{self.label} takes neutral molecules only"
        if multiplicity > 1 and self.open_shell == "no":
            return False, f"{self.label} is a closed-shell model (multiplicity {multiplicity})"
        return True, ""


_XTB_VERIFIED = ("tblite 0.8 refuses Z > 86 ('No support for elements with Z >86', checked 2026-11-09); the dxtb, "
                 "tblite and xtb engines register Z = 1-86")
_PSI4_BASIS = "Psi4 manual, appendix 'Basis Sets by Element' (the basis files Psi4 ships), read 2026-11-09"

COVERAGE: dict[str, Coverage] = {c.key: c for c in (
    Coverage("mmff94s", "MMFF94s", Z_MMFF94, open_shell="ignored", solvation="none (gas phase)",
             note="metals only as free ions (Li+, Na+, K+, Mg2+, Ca2+, Fe2+/3+, Cu+/2+, Zn2+); a molecule without "
                  "parameters is pre-optimized with UFF instead",
             source="Halgren, J. Comput. Chem. 1996, 17, 490 (MMFF94: C, H, N, O, F, Si, P, S, Cl, Br, I and those "
                    "ions)",
             verified="RDKit 2026.09 MMFFHasAllMoleculeParams on one probe molecule per element (2026-11-09): B, Se, "
                      "As, Ge, Sn, Al, Hg and Te have none", extra_elements=Z_MMFF94_IONS),
    Coverage("gfn-ff", "GFN-FF", Z_H_RN, open_shell="ignored", solvation="ALPB, GBSA (xtb, CREST)",
             source="Spicher, Grimme, Angew. Chem. Int. Ed. 2020, 59, 15665 (elements up to Rn)",
             verified="the gfnff engine registers Z = 1-86"),
    Coverage("gfn1-xtb", "GFN1-xTB", Z_H_RN, solvation="ALPB, GBSA, CPCM (tblite, xtb); none in dxtb",
             source="Grimme, Bannwarth, Shushkov, J. Chem. Theory Comput. 2017, 13, 1989 (Z = 1-86)",
             verified=_XTB_VERIFIED),
    Coverage("gfn2-xtb", "GFN2-xTB", Z_H_RN, solvation="ALPB, GBSA, CPCM (tblite, xtb); none in dxtb",
             source="Bannwarth, Ehlert, Grimme, J. Chem. Theory Comput. 2019, 15, 1652 (H to Rn)",
             verified=_XTB_VERIFIED),
    Coverage("g-xtb", "g-xTB", _z((1, 103)), solvation="limited (generalized Born, COSMO; provisional); the rank "
                                                         "step adds a GFN1-xTB CPCM term (Z = 1-86)",
             available="the rank step only (rank_method gxtb, gxtb+cpcm, auto), not a refinement engine; the "
                       "official xtb build runs it (`xtb --gxtb`)",
             source="Froitzheim, Müller, Hansen, Grimme, ChemRxiv 2025 (H to Lr, Z = 1-103); grimme-lab/g-xtb README "
                    "(preliminary version; --chrg, --uhf)",
             verified="QuantaBench ran the v2.0.1 builds for Windows, Linux and macOS arm64 (2026-11-11; the same "
                      "energies to 1e-12 Eh); elements beyond the test molecules not checked"),
    Coverage("aimnet2", "AIMNet2", Z_AIMNET2, open_shell="no", solvation="none (gas phase)",
             note="validated up to about 100 heavy atoms",
             source="Anstine, Zubatyuk, Isayev, Chem. Sci. 2025, 16, 10228; the aimnet model card (isayevlab."
                    "github.io/aimnetcentral): 14 elements, 'assumes closed-shell electronic structure'",
             verified="engines.AIMNET2_ELEMENTS (a test checks they agree)"),
    Coverage("pm7", "PM7", Z_NDDO, solvation="COSMO",
             source="openmopac.net manual, 'Elements available in PM7' (Ce-Yb only as 3+ Sparkles, left out)",
             verified="eng_mopac.ELEMENTS (a test checks they agree)"),
    Coverage("pm6-d3h4x", "PM6-D3H4X", Z_NDDO, solvation="COSMO",
             source="Stewart, J. Mol. Model. 2007, 13, 1173 (PM6: 70 elements, H-La and Lu-Bi)",
             verified="eng_mopac.ELEMENTS (a test checks they agree)"),
    Coverage("def2-svp", "def2-SVP", Z_DEF2, solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("def2-svpd", "def2-SVPD", Z_DEF2, solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("def2-tzvp", "def2-TZVP", Z_DEF2, solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("6-31g*", "6-31G*", _z((1, 36)), solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("6-31+g*", "6-31+G*", _z((1, 20)), solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("6-311+g**", "6-311+G**", _z((1, 20)), solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("aug-cc-pvdz", "aug-cc-pVDZ", _z((1, 18), (21, 36)), solvation="PCM (Psi4)", source=_PSI4_BASIS),
    Coverage("r2scan-3c", "r2SCAN-3c", Z_DEF2, solvation="PCM (Psi4)",
             available="Psi4 directly (`energy('r2scan3c/')`), not the QM stage, which takes 'functional/basis'",
             source="Grimme et al., J. Chem. Phys. 2021, 154, 064103 (def2-mTZVPP, H to Rn); " + _PSI4_BASIS),
    Coverage("crest-gfn2", "CREST (GFN2-xTB)", Z_H_RN, solvation="ALPB",
             source="as GFN2-xTB (CREST runs it; charge and spin through --chrg and --uhf)"),
    Coverage("crest-gfn1", "CREST (GFN1-xTB)", Z_H_RN, solvation="ALPB", source="as GFN1-xTB"),
    Coverage("crest-gfnff", "CREST (GFN-FF)", Z_H_RN, open_shell="ignored", solvation="ALPB", source="as GFN-FF"),
    Coverage("crest-gfn2//gfnff", "CREST (GFN2-xTB // GFN-FF)", Z_H_RN, solvation="ALPB",
             source="as GFN-FF and GFN2-xTB"),
)}


def coverage_for_engine(engine_key: str) -> Coverage | None:
    """The coverage of refinement engine ``engine_key`` (None for "auto": it picks per molecule)."""
    return COVERAGE.get(ENGINE_ENTRIES.get(engine_key, ""))


def coverage_for_crest(level: str) -> Coverage | None:
    return COVERAGE.get(CREST_LEVEL_ENTRIES.get(str(level or "").strip().lower(), ""))


def coverage_for_basis(basis: str) -> Coverage | None:
    """The coverage of a QM basis set ('def2-svp', '6-31G(d)', ...); None for a basis the table does not list."""
    e = lookup(basis)
    return COVERAGE.get(e.key) if e is not None and e.family == "Basis sets" else None


def coverage_markdown() -> str:
    """The coverage table for docs/METHODS.md."""
    rows = ["| Method | Elements | Ions | Open shells | Solvent | Source |", "|---|---|---|---|---|---|"]
    shell = {"yes": "yes", "no": "no (closed shells only)", "ignored": "spin not modelled (no electrons)"}
    for c in COVERAGE.values():
        elements = element_text(c.elements) + (f" (+ ions {element_text(c.extra_elements)})" if c.extra_elements
                                               else "")
        name = f"[{_escape(c.label, 'md')}](#{ENTRIES[c.key].anchor})"
        if c.available:
            name += f" ({_escape(c.available, 'md')})"
        src = _escape(c.source + (f"; checked: {c.verified}" if c.verified else ""), "md")
        rows.append(f"| {name} | {elements} | {'yes' if c.charged else 'no'} | {shell[c.open_shell]} | "
                    f"{_escape(c.solvation, 'md')} | {src} |")
    return "\n".join(rows)


# ============================================================================== lookup
def normalize(name: str) -> str:
    """Loose form of a name: lower case, ω/Ω -> w, no spaces, hyphens, underscores, commas or parentheses.
    '*', '+' and '/' stay: 6-31G and 6-31G* are different basis sets."""
    text = unicodedata.normalize("NFKC", str(name or "")).strip().lower()
    text = text.replace("ω", "w").replace("Ω", "w").replace("²", "2")
    return re.sub(r"[\s\-_,()]+", "", text)


def _index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for e in ENTRIES.values():
        for name in (e.key, e.anchor, e.name, *e.aliases):
            idx.setdefault(normalize(name), e.key)
    for table in (ENGINE_ENTRIES, CONFORMER_SEARCH_ENTRIES, OPTIMIZER_ENTRIES):
        for name, key in table.items():
            idx.setdefault(normalize(name), key)
    return idx


_INDEX = _index()


def get(key: str) -> Entry:
    """The entry with this exact key (KeyError otherwise)."""
    return ENTRIES[key]


def lookup(name: str) -> Entry | None:
    """The entry for ``name`` (key, name, alias or engine key; case, hyphens, spaces and ω/w do not matter)."""
    if name in ENTRIES:
        return ENTRIES[name]
    key = _INDEX.get(normalize(name))
    if key is None:
        key = next((e.key for e in ENTRIES.values() if e.anchor == str(name).strip().lower()), None)
    return ENTRIES.get(key) if key else None


def lookup_level(level: str) -> list[Entry]:
    """Entries for a 'functional/basis' level such as 'wB97X-D/def2-SVP' (the functional's, then the basis set's);
    the whole string first, so 'b3lyp/6-311+g**' finds Spartan's level and 'gfn2//gfnff' the CREST level."""
    whole = lookup(level)
    if whole is not None:
        return [whole]
    parts = [p for p in str(level).split("/") if p.strip()]
    found = [lookup(p) for p in parts]
    return [e for e in found if e is not None] if len(parts) == 2 else []


def search(text: str) -> list[Entry]:
    """Entries matching ``text``: an exact name first; else the entries whose name or aliases contain it; else the
    entries whose text contains every word of it (case-insensitive)."""
    text = str(text or "").strip()
    if not text:
        return list(ENTRIES.values())
    hit = lookup(text)
    if hit is not None:
        return [hit]
    norm = normalize(text)
    by_name = [e for e in ENTRIES.values()
               if norm and any(norm in normalize(n) for n in (e.key, e.name, *e.aliases))]
    if by_name:
        return by_name
    words = [w.lower() for w in re.split(r"\s+", text) if w]
    return [e for e in ENTRIES.values() if all(w in _haystack(e) for w in words)]


def _haystack(e: Entry) -> str:
    fields = (e.key, e.name, e.family, e.summary, e.what, e.good_for, e.watch_out, e.cost, e.dispersion,
              e.solvation, e.corrects, e.use_for, e.avoid_for, " ".join(e.aliases),
              " ".join(t for _k, t in e.implementations))
    text = " ".join(plain(f) for f in fields).lower()
    return text + " " + text.replace("ω", "w")


def suggestions(name: str, n: int = 5) -> list[str]:
    """Entry keys whose names are close to ``name`` (for 'did you mean')."""
    keys = difflib.get_close_matches(normalize(name), list(_INDEX), n=n * 2, cutoff=0.6)
    out: list[str] = []
    for k in keys:
        if _INDEX[k] not in out:
            out.append(_INDEX[k])
    return out[:n]


def for_engine(engine_key: str) -> Entry | None:
    key = ENGINE_ENTRIES.get(engine_key)
    return ENTRIES.get(key) if key else None


def engine_note(engine_key: str) -> str:
    """One line on how ``engine_key`` runs its method ('' when the guide has none)."""
    e = for_engine(engine_key)
    return e.implementation(engine_key) if e is not None else ""


def solvation_entry(engine_key: str, solvent: str, models: tuple[str, ...] = ()) -> Entry:
    """The guide entry for what ``engine_key`` does with ``solvent`` ("" = gas phase): its default model (the first of
    ``models``, its ``EngineInfo.solvation``) when it has one, the gas phase when it has none, and the general entry
    for "auto" (the model depends on the engine auto takes)."""
    if not solvent:
        return ENTRIES["gas-phase"]
    if engine_key == "auto":
        return ENTRIES["solvation"]
    if not models:
        return ENTRIES["gas-phase"]
    return ENTRIES.get(SOLVATION_MODEL_ENTRIES.get(models[0], ""), ENTRIES["solvation"])


def by_family(entries=None) -> list[tuple[str, list[Entry]]]:
    entries = list(ENTRIES.values()) if entries is None else list(entries)
    return [(f, [e for e in entries if e.family == f]) for f in FAMILIES if any(e.family == f for e in entries)]


# ============================================================================== rendering
_MARKUP = re.compile(r"`([^`]+)`|\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _render(text: str, mode: str) -> str:
    """``text`` with its markup rendered for ``mode``: "plain", "html" or "md"."""
    out, pos = [], 0
    for m in _MARKUP.finditer(text):
        out.append(_escape(text[pos:m.start()], mode))
        code, key, label = m.groups()
        if code is not None:
            out.append(f"<code>{_html.escape(code)}</code>" if mode == "html" else f"`{code}`" if mode == "md"
                       else code)
        else:
            target = ENTRIES.get(key)
            shown = label or (target.name if target else key)
            if mode == "html" and target is not None:
                out.append(f'<a href="guide:{target.anchor}">{_html.escape(shown)}</a>')
            elif mode == "md" and target is not None:
                out.append(f"[{_escape(shown, 'md')}](#{target.anchor})")
            else:
                out.append(_escape(shown, mode))
        pos = m.end()
    out.append(_escape(text[pos:], mode))
    return "".join(out)


def _escape(text: str, mode: str) -> str:
    if mode == "html":
        return _html.escape(text, quote=False)
    if mode == "md":
        return re.sub(r"([*_<>\[\]])", r"\\\1", text)
    return text


def plain(text: str) -> str:
    return _render(text, "plain")


_FIELDS = (("What it is", "what"), ("Good for", "good_for"), ("Watch out", "watch_out"), ("Cost", "cost"),
           ("Dispersion", "dispersion"), ("Solvation", "solvation"))
#: the one-line decision fields (``DECISIONS``), shown after "What it is" when the entry has them
DECISION_FIELDS = (("Corrects", "corrects"), ("Use for", "use_for"), ("Avoid for", "avoid_for"))


def _decisions(e: Entry) -> list[tuple[str, str]]:
    return [(label, getattr(e, attr)) for label, attr in DECISION_FIELDS if getattr(e, attr)]


def text_entry(e: Entry, width: int = 100) -> str:
    """The entry as plain text (``quantabench-cli methods NAME``)."""
    def para(label: str, body: str) -> list[str]:
        return textwrap.wrap(f"{label}: {plain(body)}", width, subsequent_indent="  ", break_long_words=False,
                             break_on_hyphens=False)

    lines = [f"{e.name}  [{e.family}]", *textwrap.wrap(plain(e.summary), width), ""]
    for label, attr in _FIELDS:
        lines += para(label, getattr(e, attr))
        if attr == "what":
            for d_label, body in _decisions(e):
                lines += para(d_label, body)
    if e.implementations:
        lines.append("In QuantaBench:")
        for k, note in e.implementations:
            lines += textwrap.wrap(f"{k}: {plain(note)}", width, initial_indent="  ", subsequent_indent="    ",
                                   break_long_words=False, break_on_hyphens=False)
    if e.see_also:
        lines += textwrap.wrap("See also: " + ", ".join(f"{ENTRIES[k].name} ({k})" for k in e.see_also), width,
                               subsequent_indent="  ", break_on_hyphens=False)
    if e.references:
        lines.append("References:")
        for i, r in enumerate(e.references, 1):
            lines += textwrap.wrap(f"{i}. {r.citation()} {r.url}", width, initial_indent="  ", subsequent_indent="     ",
                                   break_long_words=False, break_on_hyphens=False)
    if e.aliases:
        lines += textwrap.wrap("Also found as: " + ", ".join(e.aliases), width, subsequent_indent="  ",
                               break_on_hyphens=False)
    return "\n".join(lines)


def text_list(width: int = 100) -> str:
    """Every entry by family, one line each (``quantabench-cli methods``)."""
    keyw = max(len(k) for k in ENTRIES) + 2
    lines = []
    for family, entries in by_family():
        lines += ["", family]
        for e in entries:
            lines += textwrap.wrap(plain(e.summary), width, initial_indent=f"  {e.key:<{keyw}}",
                                   subsequent_indent=" " * (keyw + 2), break_long_words=False,
                                   break_on_hyphens=False)
    lines += ["", "Details: quantabench-cli methods NAME (e.g. methods wb97x-d, methods gfn2, methods dispersion)."]
    return "\n".join(lines[1:])


def summary_html(e: Entry, more: bool = True) -> str:
    """'<b>Name</b>: summary  More…' for the GUI's info text; the link opens the entry in the methods guide."""
    link = f' <a href="guide:{e.anchor}">More…</a>' if more else ""
    return f"<b>{_html.escape(e.name)}</b>: {_render(e.summary, 'html')}{link}"


def html_entry(e: Entry) -> str:
    """The entry as HTML for the methods guide window (QTextBrowser's HTML subset)."""
    parts = [f"<h2>{_html.escape(e.name)}</h2>",
             f'<p style="color: gray;">{_html.escape(e.family)}</p>',
             f"<p><b>{_render(e.summary, 'html')}</b></p>",
             f"<p>{_render(e.what, 'html')}</p>"]
    for label, body in _decisions(e):
        parts.append(f"<p><b>{label}:</b> {_render(body, 'html')}</p>")
    for label, attr in _FIELDS[1:]:
        parts.append(f"<p><b>{label}.</b> {_render(getattr(e, attr), 'html')}</p>")
    if e.implementations:
        items = "".join(f"<li><code>{_html.escape(k)}</code>: {_render(note, 'html')}</li>"
                        for k, note in e.implementations)
        parts.append(f"<p><b>In QuantaBench</b></p><ul>{items}</ul>")
    if e.see_also:
        links = ", ".join(f'<a href="guide:{ENTRIES[k].anchor}">{_html.escape(ENTRIES[k].name)}</a>'
                          for k in e.see_also)
        parts.append(f"<p><b>See also:</b> {links}</p>")
    if e.references:
        items = "".join(f'<li>{_html.escape(r.citation())} <a href="{_html.escape(r.url)}">doi:{_html.escape(r.doi)}'
                        f"</a></li>" for r in e.references)
        parts.append(f"<p><b>References</b></p><ol>{items}</ol>")
    if e.aliases:
        parts.append(f'<p style="color: gray;">Also found as: {_html.escape(", ".join(e.aliases))}</p>')
    return "\n".join(parts)


def html_index(entries=None, heading: str = "Methods guide") -> str:
    """Every entry (or ``entries``) by family with its summary, each name a link."""
    parts = [f"<h2>{_html.escape(heading)}</h2>"]
    groups = by_family(entries)
    if not groups:
        parts.append("<p>No entry matches.</p>")
    for family, items in groups:
        rows = "".join(f'<li><a href="guide:{e.anchor}">{_html.escape(e.name)}</a>: {_render(e.summary, "html")}</li>'
                       for e in items)
        parts.append(f"<h3>{_html.escape(family)}</h3><ul>{rows}</ul>")
    return "\n".join(parts)


def by_anchor(anchor: str) -> Entry | None:
    return next((e for e in ENTRIES.values() if e.anchor == anchor), None)


def _md_wrap(text: str, width: int = 110, indent: str = "") -> str:
    return "\n".join(textwrap.wrap(text, width, subsequent_indent=indent, break_long_words=False,
                                   break_on_hyphens=False))


def markdown_entry(e: Entry) -> str:
    lines = [f'<a id="{e.anchor}"></a>', "", f"### {_escape(e.name, 'md')}", "",
             _md_wrap(f"> {_render(e.summary, 'md')}", indent="> "), "", _md_wrap(_render(e.what, "md")), ""]
    for label, body in _decisions(e):
        lines.append(_md_wrap(f"- **{label}:** {_render(body, 'md')}", indent="  "))
    for label, attr in _FIELDS[1:]:
        lines.append(_md_wrap(f"- **{label}:** {_render(getattr(e, attr), 'md')}", indent="  "))
    if e.implementations:
        lines.append("- **In QuantaBench:**")
        for k, note in e.implementations:
            lines.append(_md_wrap(f"  - `{k}`: {_render(note, 'md')}", indent="    "))
    if e.see_also:
        links = ", ".join(f"[{_escape(ENTRIES[k].name, 'md')}](#{ENTRIES[k].anchor})" for k in e.see_also)
        lines.append(_md_wrap(f"- **See also:** {links}", indent="  "))
    if e.references:
        lines.append("- **References:**")
        for r in e.references:
            doi = _escape(r.doi, "md")
            lines.append(_md_wrap(f"  - {_escape(r.citation(), 'md')} [doi:{doi}]({r.url})", indent="    "))
    if e.aliases:
        lines.append(_md_wrap("- **Also found as:** " + ", ".join(f"`{a}`" for a in e.aliases), indent="  "))
    return "\n".join(lines)


DOC_HEADER = """\
# Methods guide

What each method, conformer search, functional, basis set and solvent model in QuantaBench is, what it is good
and bad at, and what it costs. The same text is in the app (the info under each pipeline choice, and
Help > Methods guide...) and on the command line (`quantabench-cli methods`, `quantabench-cli methods wb97x-d`).

QuantaBench does not try to reproduce Spartan exactly: its default levels include a dispersion correction, and
several of the Spartan results it is validated against (B3LYP, PM6) have none. The Spartan entries say what those
numbers are. Numbers marked "QuantaBench measured" are from [VALIDATION.md](VALIDATION.md),
[ENGINES.md](ENGINES.md) and [dev/BACKLOG.md](dev/BACKLOG.md); references were checked against Crossref.

Which method to choose for which molecules, and the review QuantaBench makes of a batch before it runs, are in
[CHOOSING_METHODS.md](CHOOSING_METHODS.md).

This file is generated from `src/quantabench/chem/method_guide.py` by `python scripts/write_methods_doc.py`.
Edit the data there, not this file (a test checks that the two agree).
"""

COVERAGE_HEADER = """\
## Which molecules each method takes

Elements, charge and spin each method, basis set and CREST level can handle (`method_guide.COVERAGE`). The review
before a run (`quantabench-cli review`, the Review window) never recommends a method for a molecule outside it.
"""


def markdown_document() -> str:
    """docs/METHODS.md."""
    toc = []
    for family, entries in by_family():
        names = ", ".join(f"[{_escape(e.name, 'md')}](#{e.anchor})" for e in entries)
        toc.append(_md_wrap(f"- **{family}:** {names}", indent="  "))
    parts = [DOC_HEADER, "## Contents", "", "\n".join(toc), "", COVERAGE_HEADER, coverage_markdown(), ""]
    for family, entries in by_family():
        parts += [f"## {family}", ""]
        for e in entries:
            parts += [markdown_entry(e), ""]
    return "\n".join(parts).rstrip() + "\n"
