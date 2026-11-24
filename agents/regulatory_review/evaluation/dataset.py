"""Golden evaluation dataset for the regulatory document review system.

Each example pairs a regulatory document with hand-authored "golden" reference
notes describing the issues a thorough, expert-level review is expected to
surface (per reviewer dimension) plus the expected overall verdict. These
references are used by the LLM-as-judge evaluators in `evaluators.py`, not for
exact string matching.

Run `python -m agents.regulatory_review.evaluation.runner` (or
`upload_dataset()` below) to push/sync this dataset to a LangSmith workspace
as a dataset named by DATASET_NAME in runner.py.
"""

from pathlib import Path

SAMPLE_DOCUMENTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sample_documents"


def _load(filename: str) -> str:
    return (SAMPLE_DOCUMENTS_DIR / filename).read_text(encoding="utf-8")


EXAMPLES = [
    {
        "inputs": {"document": _load("sample_clinical_protocol.txt")},
        "outputs": {
            "reference_notes": (
                "COMPLETENESS: Missing Scope section, missing Risk Assessment "
                "section, missing Approval/Sign-off section. Methodology "
                "present but lacks randomization/blinding detail. Safety Data "
                "present but lacks a detailed monitoring plan/stopping "
                "criteria.\n"
                "COMPLIANCE: Unsupported efficacy/safety claims ('good "
                "tolerability', 'acceptable safety profile') without "
                "supporting data/citations (ICH E6(R2)); missing disclosure "
                "of infection risk tied to the immunomodulatory mechanism; "
                "exclusion criteria omit history of malignancy/autoimmune "
                "disease; no mandatory pregnancy testing specified; no "
                "statistical method for handling missing data.\n"
                "CLARITY: Undefined acronyms on first use (PASI, DLQI, IGA, "
                "BSA); inconsistent terminology for efficacy endpoints; dense "
                "technical jargon without explanation (e.g., two-sided alpha "
                "of 0.05); 'serious infections' not defined.\n"
                "SAFETY: No detail on adverse-event management/dose "
                "adjustment or discontinuation criteria; no mention of "
                "Phase I observed AEs; monitoring limited to routine labs and "
                "infection signs only.\n"
                "EXPECTED VERDICT: Approve with changes."
            ),
            "expected_verdict": "approve with changes",
        },
    },
    {
        "inputs": {"document": _load("sample_complete_protocol.txt")},
        "outputs": {
            "reference_notes": (
                "This is a well-formed protocol synopsis intended as a "
                "'near-complete' positive control. "
                "COMPLETENESS: All required sections are present (Title, "
                "Scope, Objectives, Methodology, Risk Assessment, Safety "
                "Data, References, Approval/Sign-off) with reasonable detail "
                "including randomization/blinding methodology and a DSMB.\n"
                "COMPLIANCE: Efficacy/safety claims are supported by a "
                "cited Phase II study (GSK-4410-204); known class risks "
                "(thyroid C-cell tumors, pancreatitis) are disclosed with "
                "appropriate caveats about human relevance; pregnancy "
                "testing and contraception requirements are specified; "
                "missing-data handling (MMRM, multiple imputation) and "
                "multiplicity control are specified. Reviewers may still "
                "flag minor items such as lack of explicit dose-adjustment "
                "detail for hepatic impairment, but overall this document "
                "should score materially better on compliance/completeness "
                "than a document with major gaps.\n"
                "CLARITY: Acronyms are generally defined at first use "
                "(HbA1c, IWRS, DSMB); a few secondary acronyms (CGM, MMRM) "
                "are used without expansion and could be flagged as minor.\n"
                "SAFETY: Risk assessment explicitly discusses mechanism-based "
                "theoretical risks with mitigation, monitoring (calcitonin, "
                "solicited GI AEs), stopping rules, and DSMB oversight.\n"
                "EXPECTED VERDICT: Approve (or Approve with minor changes)."
            ),
            "expected_verdict": "approve",
        },
    },
    {
        "inputs": {"document": _load("sample_investigator_brochure_excerpt.txt")},
        "outputs": {
            "reference_notes": (
                "This document is deliberately poor and should receive a "
                "strongly negative review. "
                "COMPLIANCE: Promotional, unsupported marketing-style claims "
                "('works really well', 'basically safe', 'better than "
                "competitor products', 'Patients love it') are inappropriate "
                "for an Investigator's Brochure and violate ICH E6(R2) "
                "standards for objective, data-driven safety communication. "
                "The known JAK-inhibitor class boxed-warning risks (serious "
                "infection, MACE, VTE, malignancy, thrombosis) are named but "
                "not quantified with trial incidence data, and no monitoring "
                "or dose-adjustment guidance is given despite being "
                "standard-of-care for this drug class.\n"
                "COMPLETENESS: Nonclinical Overview and Clinical Experience "
                "sections are placeholders with no actual data ('Everything "
                "looked fine', 'data tables available upon request'); no "
                "References section; no sign-off/approval section.\n"
                "CLARITY: Vague, non-specific statements throughout "
                "('Investigators should use clinical judgement') provide no "
                "actionable guidance.\n"
                "SAFETY: Does not quantify observed AE incidence, does not "
                "describe required monitoring (lipid panel, lymphocyte "
                "counts) or hepatic/renal dose adjustments for a boxed-"
                "warning drug class.\n"
                "EXPECTED VERDICT: Reject (requires substantial rewrite "
                "before resubmission)."
            ),
            "expected_verdict": "reject",
        },
    },
    {
        "inputs": {"document": _load("sample_package_insert_draft.txt")},
        "outputs": {
            "reference_notes": (
                "COMPLETENESS: Core prescribing-information sections are "
                "present (Indications, Dosage, Contraindications, Warnings, "
                "Adverse Reactions, Drug Interactions, Use in Specific "
                "Populations, Clinical Studies) but several are explicitly "
                "incomplete ('not included in this excerpt', 'not further "
                "described'), notably the renal dosing table and full drug-"
                "interaction list.\n"
                "CLARITY: Uses non-standard abbreviations/shorthand ('tx', "
                "'pts', 'CV', '50mg') inappropriate for formal prescribing "
                "information; inconsistent spacing/formatting of dose units.\n"
                "COMPLIANCE: Adverse reaction incidence is stated only as "
                '">= 5%" without comparator rates or a table, which is '
                "insufficient for FDA/EMA labeling standards; geriatric-use "
                "statement claims no safety/efficacy differences without "
                "citing supporting subgroup data; the pivotal trial result "
                "is referenced without the actual hazard ratio or p-value.\n"
                "SAFETY: Hyperkalemia/hypotension/renal monitoring are "
                "mentioned but lack specific thresholds, periodicity, or "
                "stopping/action criteria; CYP3A4 interaction is flagged as "
                "a contraindication without explaining the mechanism or "
                "risk.\n"
                "EXPECTED VERDICT: Approve with changes (major revisions "
                "needed before the excerpt could be finalized)."
            ),
            "expected_verdict": "approve with changes",
        },
    },
    {
        "inputs": {"document": _load("sample_psur_excerpt.txt")},
        "outputs": {
            "reference_notes": (
                "This is a well-structured, compliant Periodic Safety Update "
                "Report executive summary and should score favorably. "
                "COMPLETENESS: Standard ICH E2C(R2) PSUR sections are all "
                "present: introduction, marketing authorization status, "
                "actions taken, RSI changes, exposure estimates, case "
                "presentation, signal evaluation, benefit-risk analysis, "
                "conclusion, references, and approval/sign-off.\n"
                "COMPLIANCE: Identified risks (hypersensitivity) are "
                "correctly cross-referenced to the Risk Management Plan; "
                "signal evaluation process is described; benefit-risk "
                "conclusion is stated with rationale; cites ICH E2C(R2) "
                "appropriately.\n"
                "CLARITY: Terminology is consistent and standard for "
                "pharmacovigilance documents (SAE, CCDS, RMP all used "
                "correctly); minor opportunity to spell out acronyms like "
                "CCDS on first use for a general audience.\n"
                "SAFETY: Reports concrete counts (410 SAEs, 18 "
                "hypersensitivity cases, 6 fatal cases with causality "
                "assessment) and an evaluated new signal (injection-site "
                "nodules) with a documented monitoring decision, which is "
                "appropriately detailed safety reporting.\n"
                "EXPECTED VERDICT: Approve."
            ),
            "expected_verdict": "approve",
        },
    },
]


def get_dataset_examples() -> list[dict]:
    """Return the list of {inputs, outputs} example dicts."""
    return EXAMPLES
