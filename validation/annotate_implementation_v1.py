"""Create a separate bounded implementation review; never edit original TeX."""
import hashlib
import json
from pathlib import Path

import oracle_config as oc

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]/'trajectory_manifold_implementation.tex'
OUTPUT = HERE/'reviews/implementation_tex_v1'

# Anchor, ID, category, annotation. Comments address actual document claims.
NOTES = [
    ('No consistency, optimizer convergence, or closed-loop stability claim is made here.', 'R01', 'SCOPE / VERIFIED',
     r'The text explicitly disclaims these guarantees. The validation results must not be presented as disproving guarantees the authors did not claim. This review checks code correspondence and elementary structure; a complete proof, citation, and novelty audit remains open.'),
    (r'Its canonical exact encoder and decoder are $E^\star=\pi|_{\mathcal B_N^+}$ and $D^\star=\Phi_N$.', 'R02', 'MATHEMATICAL QUALIFICATION',
     r'The deterministic rollout is a graph over $(x_0,\mathbf u)$. The chosen coordinate domain and smoothness hypotheses matter for formal manifold statements: an open smooth domain gives the usual graph result; closed input boxes introduce boundaries/corners, and safety-restricted sets need not be manifolds. No theorem about the supplied Part I paper is certified in this bounded review.'),
    ('before extracting transitions.', 'R03', 'VERIFIED / LIMITATION',
     r'The runner splits whole episodes before flattening transitions and fits normalization/local regression on training data. This avoids this particular train/test transition leak. It does not establish independent coverage of closed-loop states or an effective sample size equal to the transition count; transitions within episodes are dependent.'),
    ('not a prescribed physical transition.', 'R04', 'VERIFIED',
     r'\code{DataDrivenTransition} matches the residual standardized SiLU construction. No pendulum kinematic equation is inserted into that network. Plant-specific simulator and unit choices still enter collection/evaluation through the runner, so ``system agnostic architecture'' should not be expanded into ``no plant-specific harness assumptions.'''),
    (r'The learned map is not explicitly constrained to satisfy $\widehat f_\vartheta(0,0)=0$.', 'R05', 'ACCURATE LIMITATION / OPEN TEST',
     r'The one-step normalized loss, AdamW schedule, validation checkpointing and absent equilibrium constraint match the active code. Neither free-run accuracy nor equilibrium/derivative fidelity follows from one-step loss alone. Full-budget H/E learning and uncertainty-calibration evidence remains open; oracle Gate A is not that evidence.'),
    ('no additional time parameterization is learned.', 'R06', 'VERIFIED / POSSIBLE UNDERSTATEMENT',
     r'Because the output copies every input and $x_0$, projection identity implies injectivity. For a smooth open-domain decoder, its Jacobian contains an identity block and has full column rank, and projection is a continuous inverse on its graph. These structural facts hold even for inaccurate learned dynamics. At $N=30$ the full chart has dimension 34; the fixed-state online search has dimension 30, as in ordinary single shooting. They do not by themselves demonstrate compression, data efficiency or novelty.'),
    ('not its first input to the previously applied input.', 'R07', 'VERIFIED / REQUIREMENT DISTINCTION',
     r'The objective and input-rate indexing match \code{DataDrivenCEMSolver._cost}. The submitted runner specifies normalized input bounds $[-10,10]$; those were not invented by validation. All-time cart rail $\pm2.4$ m and cart recentering were reviewer-added requirements. A miss under those added requirements is a reviewer-task outcome, not a violation of an authored hard-state constraint.'),
    ('Gradients are not used in this active optimizer.', 'R08', 'VERIFIED / FINITE-SEARCH LIMIT',
     r'Cost-only elite selection, unbiased elite standard deviation, smoothing, clipping, current-mean slot and best sampled incumbent match the active path. Three finite iterations do not certify a constrained/global optimum or successful recovery; the text correctly avoids a global-minimizer claim. The completed selected CEM diagnosis improved cost in 22/40 state/warm pairs without changing combined predicted safety (18/40). Those are selected 1.2-second oracle predictions, not a trained-H result or an attainability certificate.'),
    (r'This ridge regression has no intercept.', 'R09', 'VERIFIED / IDENTIFICATION QUESTION',
     r'Nearest-6000 selection uses population-standardized state distance only, and the fitted linear map has no intercept, as stated. Large selected inputs, finite neighborhood size, excitation/conditioning and nonlinear regression bias still need measurement. This fitted map is not automatically the plant Jacobian at upright. The text makes no such equality claim.'),
    (r'the active MPC independently uses $Q_f=12Q$.', 'R10', 'VERIFIED / STABILITY LIMIT',
     r'Riccati $P$ is separate from the predictive terminal penalty. DARE stabilizes the fitted linear model under its usual hypotheses; it is not a nonlinear saturated/hybrid stability proof. The original oracle local gain has slow cart centering under $Q_p=.02$, but slow centering is not an authored hard recentering deadline. Selected $Q_p=2$ local-only probes do not certify the proposed hybrid entry region.'),
    ('resets the stored plan to zero.', 'R11', 'VERIFIED / LOCAL-REGION LIMIT',
     r'The strict switch $x^TQx<.75$, queue ordering and lack of hysteresis match the submitted wrapper. For a pure cart displacement this switch permits $|p|<\sqrt{.75/.02}\simeq6.12$ m. That explains why a reviewer rail/recentering task can disagree with the controller objective; it is not proof of an unsafe design relative to unspecified rails. No nonlinear invariant region is established.'),
    ('This describes the current queue implementation exactly.', 'R12', 'VERIFIED / IMPLEMENTATION CONCERN',
     r'The code does retain the old plan across local intervals and always shifts by five, rather than by the actual elapsed physical samples. This is a concrete stale-warm-start concern; the specification accurately reports it. Its quantitative effect on closed-loop performance has not been isolated. OA-1 is a separate proposed oracle amendment and does not repair or change submitted H.'),
    (r'\code{controller_lr} is unused by the active CEM path.', 'R13', 'VERIFIED / CONFIGURATION SCOPE',
     r'The dev-profile table and its 30-step/0.6-second horizon match the submitted runner. Completed validation Gate A instead used a true-model 60-step horizon, float64 reference and 20-second development cases. The existing 29 tests and synthetic CPU/CUDA equivalence passed, conditional on the disclosed unused-CVXPY-import workaround; same-author checks do not replace independent author review.'),
    ('the initial sampling box does not constrain later states.', 'R14', 'VERIFIED / RANGE REPORTING',
     r'The collection box is in the submitted runner; validation reused it for a different development stress distribution. The modulo hold rule and lack of state truncation are accurately stated. Report observed excursions, settling and tested recovery behavior as conditional diagnostics. Neither a collection box nor finitely many trajectories identifies the maximum recoverable range or the constrained viable set.'),
    ('There is no online parameter adaptation.', 'R15', 'VERIFIED / CLAIM BOUNDARY',
     r'The active path uses a transition-composed rollout and a separately data-identified switched linear law; legacy latent/autoencoder/curvature solvers are inactive. This is computationally ordinary learned-dynamics shooting in canonical coordinates. Equivalence alone does not refute geometric interpretation or establish a literature novelty verdict. External claims of superior interpretability/data efficiency should be assessed separately unless actually made in this text.'),
    (r'\begin{thebibliography}{1}', 'R16', 'UNRESOLVED / NOT AN ERROR VERDICT',
     r'The supplied PDF transcription, original Part I hypotheses/proofs and bibliography need their own source-by-source review. No literature-backed novelty verdict or confirmed theorem/citation error is asserted here. In this bounded active-code correspondence pass, no confirmed false equation or prose/code mismatch was identified; accurate limitations and open validation needs are marked above.'),
]


def main():
    original = SOURCE.read_text()
    snapshot = HERE/'handoff/sources/trajectory_manifold_implementation.tex'
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if snapshot.read_bytes() != SOURCE.read_bytes():
        raise RuntimeError('Original and handoff transcription differ; review before annotating')
    sources = {p: hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest() for p in oc.SOURCE_HASHES}
    if sources != oc.SOURCE_HASHES:
        raise RuntimeError('Submitted source pins changed')
    rendered, inserted, ledger = original, [], []
    intro = '\n\\review{\\textbf{October 9, 2026 bounded review.} Categories distinguish verified statements, mathematical qualifications, implementation concerns, possible understatements and unresolved checks. No confirmed algebraic/code-correspondence error was found in this pass. Reviewer-added rail/recentering misses are not failures of the authors\' stated design. Gate-A endpoint revision is now requested and its exact new rule is pending. Historical strict scores remain unchanged.}\n'
    rendered = rendered.replace('\\maketitle', '\\maketitle'+intro, 1)
    inserted.append(intro)
    for anchor, ident, category, note in NOTES:
        if rendered.count(anchor) != 1:
            raise RuntimeError(f'Anchor not unique: {ident}')
        block = '\n\\review{\\textbf{'+ident+' ('+category+').} '+note+'}\n'
        if anchor.startswith('\\begin{thebibliography}'):
            rendered = rendered.replace(anchor, block+anchor, 1)
        else:
            rendered = rendered.replace(anchor, anchor+block, 1)
        inserted.append(block)
        ledger.append(dict(id=ident, category=category, anchor=anchor, annotation=note))
    restored = rendered
    for block in reversed(inserted):
        restored = restored.replace(block, '', 1)
    if restored != original:
        raise RuntimeError('Review changed original transcription prose')
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT/'trajectory_manifold_implementation_reviewed_v1.tex').open('x') as f:
        f.write(rendered)
    with (OUTPUT/'annotations.json').open('x') as f:
        json.dump(ledger, f, indent=2); f.write('\n')
    manifest = dict(source=str(SOURCE), source_sha256=source_hash, snapshot_sha256=source_hash,
                    source_prose_preserved=True, comments=len(NOTES)+1,
                    reviewed_HEAD='4c9de6db91d1c2306c0cb1318c8b309f4da390df', submitted_source_sha256=sources,
                    review_scope='bounded current implementation/code and existing evidence; same-author; no new experiments',
                    confirmed_algebraic_or_code_correspondence_errors=0,
                    complete_theory_citation_novelty_audit='open', revised_gate_a_endpoint='requested; exact rule pending',
                    pdf_compilation='not performed; pdflatex unavailable',
                    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUTPUT.iterdir())})
    with (OUTPUT/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2); f.write('\n')
    print(json.dumps({'output':str(OUTPUT),'annotations':len(NOTES)+1,'original_prose_preserved':True}))


if __name__ == '__main__':
    main()
