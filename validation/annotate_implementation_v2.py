"""Produce method-focused annotations without changing the source transcription.

The v1 builder supplies only the frozen anchor/ID list. Its annotations are
replaced completely; no scientific dependency or experiment is invoked.
"""
import hashlib
import json
from pathlib import Path

from annotate_implementation_v1 import NOTES as ANCHORS
import oracle_config as oc

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / 'trajectory_manifold_implementation.tex'
OUTPUT = HERE / 'reviews/implementation_tex_v2'
PAPER = 'https://arxiv.org/html/2609.19079v1'

NOTES = {
    'R01': ('CLAIM SCOPE', r"This specification explicitly disclaims consistency, optimizer convergence and closed-loop stability guarantees. Their absence is not a false claim. Assess its fidelity to the implementation separately from the foundational paper's mathematical claims and from demonstrated practical benefit."),
    'R02': ('MATHEMATICAL QUALIFICATION', r'The deterministic rollout is a graph over initial state and inputs. Formal smooth-manifold statements depend on the coordinate domain: an open smooth domain gives the usual graph construction; a closed input box introduces boundaries/corners, and arbitrary state restrictions need not define manifolds. The precise hypotheses and conclusions of the foundational paper still require independent review.'),
    'R03': ('CODE CORRESPONDENCE / DATA LIMIT', r'The runner splits complete episodes before flattening transitions and fits scaling and local regression on training data. That prevents this particular train/test leakage. Within-episode dependence, state-action coverage and closed-loop distribution shift still need assessment; transition count is not an independent-sample count.'),
    'R04': ('CODE CORRESPONDENCE', r'The active transition class implements this standardized residual SiLU network without inserting pendulum dynamics or kinematic identities. Plant-specific units and simulation still enter the data/evaluation harness. The generic architecture and the plant-specific harness should be distinguished.'),
    'R05': ('ACCURATE LIMITATION / EVIDENCE NEEDED', r'The active loss, optimizer, checkpoint selection and absent equilibrium constraint agree with the description. One-step loss does not establish free-run accuracy, equilibrium fidelity or derivative fidelity. Held-out rollout errors and trained-controller comparisons remain needed; no comparative performance conclusion is available from this correspondence check.'),
    'R06': ('STRUCTURAL CONSEQUENCE / POSSIBLE UNDERSTATEMENT', r'Copying all inputs and the initial state makes the decoder injective independently of model accuracy. On an open smooth domain its Jacobian contains an identity block and has full column rank; projection is a continuous inverse on its graph. At $N=30$ the full chart has dimension 34 and the fixed-state input search dimension 30. These identities should be separated from compression, fidelity and empirical data efficiency. Comparison with ordinary single shooting is a literature-review question, not a novelty verdict established here.'),
    'R07': ('CODE CORRESPONDENCE / CONSTRAINT SCOPE', r'The objective and adjacent-input rate indexing agree with the active cost. The submitted input bounds are normalized $[-10,10]$, corresponding to $\pm98.1$ N for the specified plant. State penalties are soft costs; the method does not promise hard rail limits or a cart-recentering deadline. Such additional requirements must be identified as external evaluation choices.'),
    'R08': ('CODE CORRESPONDENCE / FINITE SEARCH', r'The active optimizer uses cost-ranked elites, sample standard deviation, smoothing, clipping and the best sampled incumbent. Its finite population and three iterations do not certify a global optimum or physical recoverability. The text accurately distinguishes the numerical search result from the minimizer in (13). Search adequacy and model error need separate empirical assessment.'),
    'R09': ('CODE CORRESPONDENCE / IDENTIFICATION QUESTION', r'The nearest-6000 selection uses population-standardized state distance only, with no input-neighborhood restriction or fitted intercept. Excitation, conditioning, neighborhood size and nonlinear regression bias remain unmeasured. This fitted linear map is not automatically the plant Jacobian at upright; the text does not claim that equality.'),
    'R10': ('CODE CORRESPONDENCE / STABILITY SCOPE', r'The Riccati matrix and predictive terminal weight are correctly distinguished. A stabilizing DARE solution for a fitted linear model does not by itself certify the nonlinear saturated or switched controller. Neither an invariant switching region nor such a certificate is claimed here. The complete hybrid, including its identified local law, must be evaluated when attributing performance.'),
    'R11': ('SWITCHING QUESTION', r'The quadratic switch and queue ordering match the active wrapper. For pure cart displacement the stated weights allow $|p|<\sqrt{.75/.02}\simeq6.12$ m into local control. Thus this test is not a uniformly small physical-state neighborhood. The relevant question is observed local recovery and saturation across the declared operating domain; no nonlinear invariant region follows from this threshold.'),
    'R12': ('IMPLEMENTATION CONCERN / EFFECT UNMEASURED', r'The stored plan is retained across local-control intervals and always shifted by five rather than by elapsed plant samples. That can produce a stale warm start after an interruption. The specification accurately reports this behavior; its quantitative effect on performance has not been isolated.'),
    'R13': ('CONFIGURATION SCOPE / REPRODUCTION OPEN', r'The development profile, 30-step horizon, float32 neural computation and double-precision local fit agree with the active runner. The horizon is 0.6 physical seconds and the nominal plan application interval is 0.1 seconds. A complete original-budget reproduction and a trained baseline comparison have not been completed.'),
    'R14': ('OPERATING-RANGE QUESTION', r'The initial sampling box and absence of state truncation agree with the runner. A collection distribution does not establish a recoverable operating domain. Report tested initial conditions, transient excursions and settling separately. Neither finite trajectories nor a failed sampled search certifies the maximum recoverable set or physical impossibility of recovery.'),
    'R15': ('ACTIVE METHOD / ATTRIBUTION', r'The active computation combines a transition-composed learned rollout, sampled input optimization and switched data-identified linear feedback. Legacy latent/autoencoder/curvature routines are inactive. Computational correspondence to learned-dynamics shooting does not settle the value or novelty of the geometric formulation. Attribute empirical gains to the complete method unless component comparisons establish more.'),
    'R16': ('FOUNDATIONAL PAPER / REVIEW PENDING', r"Claude's review must include Arda Bayer's exact arXiv v1 submission, \url{https://arxiv.org/html/2609.19079v1}, as the foundation of this implementation. Check its actual claims, assumptions, proofs and references, then map them to this specification. A source-by-source mathematical audit and literature comparison remain open. This bounded code-correspondence pass identified no confirmed false equation or prose/code mismatch; it is not certification of the paper or a novelty verdict."),
}


def main():
    original = SOURCE.read_text()
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if (HERE/'handoff/sources/trajectory_manifold_implementation.tex').read_bytes() != SOURCE.read_bytes():
        raise RuntimeError('Source and transcription snapshot differ')
    sources = {p: hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest() for p in oc.SOURCE_HASHES}
    if sources != oc.SOURCE_HASHES:
        raise RuntimeError('Submitted source pins changed')
    intro = '\n\\review{\\textbf{Method assessment, October 9, 2026.} Comments distinguish code correspondence, mathematical qualifications and open evidence needs. The foundational arXiv paper, literature novelty and comparative trained-controller performance still require review. This annotated copy preserves the original transcription; it is not the author\'s original TeX.}\n'
    if original.count('\\maketitle') != 1 or set(NOTES) != {n[1] for n in ANCHORS}:
        raise RuntimeError('Unexpected anchors or annotation IDs')
    rendered = original.replace('\\maketitle', '\\maketitle'+intro, 1)
    inserted, ledger = [intro], []
    for anchor, ident, _, _ in ANCHORS:
        category, note = NOTES[ident]
        if rendered.count(anchor) != 1:
            raise RuntimeError(f'Anchor not unique: {ident}')
        block = '\n\\review{\\textbf{'+ident+' ('+category+').} '+note+'}\n'
        rendered = rendered.replace(anchor, block+anchor if anchor.startswith('\\begin{thebibliography}') else anchor+block, 1)
        inserted.append(block)
        ledger.append(dict(id=ident, category=category, anchor=anchor, annotation=note))
    restored = rendered
    for block in reversed(inserted):
        restored = restored.replace(block, '', 1)
    if restored != original:
        raise RuntimeError('Original transcription prose changed')
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'trajectory_manifold_implementation_reviewed_v2.tex').write_text(rendered)
    (OUTPUT/'annotations.json').write_text(json.dumps(ledger, indent=2)+'\n')
    manifest = dict(source=str(SOURCE), source_sha256=source_hash,
                    source_prose_preserved=True, comments=len(ledger)+1,
                    reviewed_HEAD='4c9de6db91d1c2306c0cb1318c8b309f4da390df',
                    submitted_source_sha256=sources, paper_for_claude=PAPER,
                    paper_review='pending; not undertaken by this annotation pass',
                    review_scope='same-author bounded active-code correspondence and elementary mathematical qualifications',
                    literature_review='incomplete', trained_baseline_comparison='not executed',
                    pdf_compilation='not performed; pdflatex unavailable',
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir())},
                    builder_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                    [Path(__file__), HERE/'annotate_implementation_v1.py']})
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(OUTPUT), comments=len(ledger)+1, source_prose_preserved=True)))


if __name__ == '__main__':
    main()
