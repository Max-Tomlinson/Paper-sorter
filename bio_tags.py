"""bio_tags.py — Biology metadata detection for sort_papers.py.

Detects omics technologies, organism, article type, diseases,
trial status, and tissue from paper text using regex patterns.
"""

from __future__ import annotations
import re

# ── Pattern tables ─────────────────────────────────────────────────────────────

OMICS_PATTERNS: list[tuple[str, str]] = [
    ("Genomics",        r"genom|GWAS|genome.wide|WGS|whole.genome.sequenc|SNP array|exome"),
    ("Transcriptomics", r"transcriptom|RNA.seq|scRNA|gene.expression|mRNA|single.cell.RNA|bulk RNA"),
    ("Proteomics",      r"proteom|mass.spectrometry|LC.MS|LC-MS|protein.expression|TMT|iTRAQ"),
    ("Epigenomics",     r"epigenom|methylat|ATAC.seq|ChIP.seq|WGBS|histone|chromatin.access"),
    ("Metabolomics",    r"metabolom|metabolite|NMR.spectroscopy|untargeted.metabol"),
    ("Metagenomics",    r"metagenom|microbiom|16S|shotgun.sequenc|gut.microbi"),
    ("Lipidomics",      r"lipidom|lipidome"),
    ("Single-cell",     r"single.cell|scRNA|scATAC|single nucleus|snRNA|spatial.transcriptom"),
    ("Multi-omics",     r"multi.omics|multiomics|multi-omics|integrat.omics"),
]

ORGANISM_PATTERNS: list[tuple[str, str]] = [
    ("Human",      r"\bhuman\b|\bpatient[s]?\b|\bcohort\b|\bparticipant[s]?\b|\bindividual[s]?\b|\bmen\b|\bwomen\b|\bsubject[s]?\b"),
    ("Mouse",      r"\bmice\b|\bmouse\b|\bmurine\b|Mus musculus"),
    ("Rat",        r"\brat[s]?\b|\bRattus\b"),
    ("Zebrafish",  r"\bzebrafish\b|Danio rerio"),
    ("Drosophila", r"\bdrosophila\b|\bfruit fly\b|\bfruit flies\b"),
    ("C. elegans", r"C\.\s*elegans|\bnematode\b"),
    ("Primate",    r"\bprimate[s]?\b|\bmacaque\b|\bmonkey\b|\bmarmoset\b"),
    ("In vitro",   r"\bcell line[s]?\b|\bin vitro\b|\bHeLa\b|\bHEK293\b|\bin-vitro\b"),
]

ARTICLE_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("Meta-analysis",  r"meta.analysis|meta analysis|systematic review and meta"),
    ("Review",         r"\breview\b|\bwe review\b|\bthis review\b|\boverview\b|\bsurvey of\b"),
    ("Clinical Trial", r"clinical trial|randomized controlled|randomised controlled|double.blind|placebo.controlled|NCT\d{6,}"),
    ("Cohort Study",   r"cohort study|longitudinal study|prospective study|retrospective study|population.based study"),
    ("Case-Control",   r"case.control|case control"),
    ("Methods Paper",  r"we present|we introduce|we develop|we describe a (novel|new)|novel (method|tool|pipeline|algorithm|framework|software)|benchmarking|open.source (tool|software)|github\.com|R package|python package|bioconductor"),
]

DISEASE_PATTERNS: list[tuple[str, str]] = [
    ("Cancer",            r"cancer|tumor|tumour|carcinoma|malignant|oncolog|leukemia|lymphoma|glioma|melanoma|sarcoma"),
    ("Alzheimer's",       r"alzheimer|alzheimer.s disease|\bAD\b.{0,20}(dementia|disease)|dementia"),
    ("Parkinson's",       r"parkinson|parkinson.s disease|\bPD\b.{0,20}(patient|disease)"),
    ("Diabetes",          r"diabetes|diabetic|\bT2D\b|\bT1D\b|insulin resistance|hyperglycemi"),
    ("Cardiovascular",    r"cardiovascular|heart disease|coronary|myocardial|atherosclerosis|hypertension|stroke"),
    ("Neurodegeneration", r"neurodegenerat|\bALS\b|amyotrophic lateral|multiple sclerosis|\bMS\b.{0,10}patient"),
    ("Aging",             r"\baging\b|\bageing\b|age.related|longevity|lifespan|healthspan|geroscience"),
    ("Obesity",           r"\bobesity\b|\bobese\b|overweight|adiposity|BMI"),
    ("Autoimmune",        r"autoimmune|lupus|rheumatoid arthritis|\bIBD\b|crohn|ulcerative colitis"),
    ("Kidney Disease",    r"kidney disease|renal disease|\bCKD\b|chronic kidney|nephropathy"),
    ("Frailty",           r"\bfrailty\b|\bfrail\b|sarcopenia"),
    ("Infectious",        r"infection|infectious|bacterial|viral|pathogen|HIV|COVID|SARS|influenza"),
]

TISSUE_PATTERNS: list[tuple[str, str]] = [
    ("Blood",       r"\bblood\b|\bplasma\b|\bserum\b|\bPBMC\b|\bleukocyte|\berythrocyte|\bplatelet|\bwhole.blood"),
    ("Brain",       r"\bbrain\b|\bcortex\b|\bhippocampus\b|cerebrospinal|\bneuron[s]?\b|\bcerebral\b|\bprefrontal"),
    ("Liver",       r"\bliver\b|\bhepatic\b|\bhepatocyte|\bhepato"),
    ("Kidney",      r"\bkidney\b|\brenal\b|\bnephron|\btubular"),
    ("Heart",       r"\bheart\b|\bcardiac\b|\bmyocardium|\bcardiomyocyte"),
    ("Muscle",      r"\b(skeletal )?muscle\b|\bmyocyte|\bmyoblast"),
    ("Adipose",     r"\badipose\b|\bfat tissue\b|\badipocy|\bwhite fat\b|\bbrown fat\b"),
    ("Lung",        r"\blung[s]?\b|\bpulmonary\b|\balveolar\b|\bairway\b"),
    ("Skin",        r"\bskin\b|\bdermis\b|\bepidermis\b|\bfibroblast[s]?\b"),
    ("Gut",         r"\bgut\b|\bintestin|\bcolon\b|\bbowel\b|\bgastrointestinal\b|\bduodenum\b"),
    ("Bone Marrow", r"bone marrow|\bhematopoiet"),
    ("Pancreas",    r"\bpancreas\b|\bpancreatic\b|\bislet[s]?\b|\bbeta.cell[s]?\b"),
    ("Saliva",      r"\bsaliva\b|\bsalivary\b"),
    ("Urine",       r"\burine\b|\burinary\b|\burothelial"),
]

_TRIAL_RE = re.compile(
    r"clinical trial|randomized controlled|randomised controlled|\bRCT\b|NCT\d{6,}|phase (I|II|III|IV) trial",
    re.I,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def tag(title: str, abstract: str, keywords: list[str], raw_text: str) -> dict:
    """Return a dict of detected biology tags."""
    # Weight title and abstract more heavily than full text
    t = " ".join([
        title * 3,
        abstract * 2,
        " ".join(keywords),
        raw_text,
    ])
    return {
        "omics":        _match_all(t, OMICS_PATTERNS),
        "organism":     _match_first(t, ORGANISM_PATTERNS),
        "article_type": _match_first(t, ARTICLE_TYPE_PATTERNS) or "Research Article",
        "diseases":     _match_all(t, DISEASE_PATTERNS),
        "is_trial":     bool(_TRIAL_RE.search(t)),
        "tissues":      _match_all(t, TISSUE_PATTERNS),
    }


def _match_all(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    return [label for label, pat in patterns if re.search(pat, text, re.I)]


def _match_first(text: str, patterns: list[tuple[str, str]]) -> str:
    for label, pat in patterns:
        if re.search(pat, text, re.I):
            return label
    return ""
