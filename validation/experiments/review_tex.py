"""Produce v3 reviewer comments while preserving the source transcription."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT.parent/'trajectory_manifold_implementation.tex'
out=ROOT/'validation/results_v1/reviews/implementation_tex_v3'
out.mkdir(parents=True,exist_ok=True)
original=source.read_text()
notes=json.loads((ROOT/'validation/reviews/implementation_tex_v2/annotations.json').read_text())
revisions={
'R02':r'The pinned Part I graph proof is correct on an open admissible coordinate domain. The closed input box gives boundaries/corners; arbitrary state constraints need extra regularity. This note imports $\mathcal D_N$ without defining its admissibility conditions locally.',
'R09':r'The nearest-6000 rule selects by standardized state distance only. Inputs remain unrestricted, and it uses all training triples when $L\leq6000$. The selected radius and fit RMSE are measured in E0/E2. This is a neighborhood regression, not necessarily the derivative at upright.',
'R13':r'The configuration agrees with the active runner. A fresh unchanged seed-234 dev run reproduces angular balance at all six authored initial angles. Multi-seed comparisons are saved under validation/results_v1/E0. This establishes a demonstration, not a guaranteed operating region.',
'R16':r'Part I Theorem 3.3, Corollaries 4.1/4.3/4.4 and the sampled-map construction pass the bounded proof audit under their stated smoothness/domain assumptions. They establish a graph and coordinate dimension, not learned-model accuracy or closed-loop performance. The complete audit and primary-source comparison are in validation/results_v1/reviews/paper_review_v1. No equation/code contradiction was found; the additional comments identify specification omissions and qualifications.',
}
for item in notes:
    if item['id'] in revisions:item['annotation']=revisions[item['id']]
notes[-1]['category']='FOUNDATIONAL PAPER / REVIEW COMPLETED'
extra=[
('R17','NOTATION',r'Below, $\vartheta$ denotes network parameters;',r'Use distinct symbols for episode count $D$ and decoder $D^\star$, dataset $\mathcal D$ and domain $\mathcal D_N$, transition index $j$ and CEM iteration $j$, network $h_\vartheta$ and hold integer $h$, position $p$ and candidate index $p$, activation argument $a$ and lower input bound $a$, and length $l$ versus cost $\ell_{QR}$. Subscripts disambiguate the several $\mu$ values, but a notation table would help.'),
('R18','REFERENCE / NORMALIZATION',r'For fixed references $x_{\rm ref},u_{\rm ref}$, define',r'The runner uses $x_{\rm ref}=0$ and $u_{\rm ref}=0$. $Q,R,Q_f$ weight the normalized coordinates of \eqref{eq:units}; they are not weights directly on physical velocities and force. The switch $x^\top Qx$ assumes the same zero reference. $Q$ is defined separately for local fitting and CEM in the runner; the current values agree.'),
('R19','CROSS REFERENCES',r'The identity term in (6)',r'Replace hard-coded equation numbers such as (6) and (13) with \texttt{\string\eqref} using the existing equation labels, so subsequent edits do not break references.'),
('R20','SCHEDULE QUALIFICATION',r'a cosine learning-rate schedule ending at zero over the configured epoch count.',r'The scheduler steps after each epoch with $T_{\max}=\text{epochs}$. It reaches zero after the final optimizer epoch; the last epoch itself trains at a small positive learning rate. Thus an endpoint of zero describes the scheduler state, not the last training updates.'),
('R21','COLLECTION DEAD DRAW',r'This is the actual modulo-based update rule, rather than a countdown hold schedule.',r'At $k=0$, $0\bmod h=0$ for every sampled positive $h$: the initially drawn hold is overwritten before use. The initial input affects even episodes through the noisy update; odd episodes redraw it before any plant step. The unused hold draw still advances the RNG and therefore cannot simply be removed in a reproduction.'),
('R22','SWITCH GEOMETRY / PRECISION',r'There is no switching hysteresis or certified invariant switching region.',r'The axis intercepts are $|p|<6.12$ m, physical $|\dot p|<0.495$ m/s, $|\theta|<0.0866$ rad, and physical $|\dot\theta|<0.858$ rad/s, each with the other coordinates zero. They are not independent rectangular bounds. Position can exceed the initial collection box. $Q$ is stored in float32; the NumPy switch with float64 simulated state promotes the product to float64, while local fitting/DARE is float64 and CEM is float32.'),
('R23','NONFINITE CEM EDGE',r'A nonfinite reported CEM cost causes a zero input and resets the stored plan to zero.',r'The implementation applies argmin to costs without first masking NaNs. A NaN-selected index can hide a finite candidate for that iteration. If no finite incumbent is accepted, the best plan remains the warm-start mean and reported finite is false. The wrapper then applies the disclosed zero-input fallback.'),
('R24','EVALUATION OMISSION',r'The current \code{dev} profile uses the following settings.',r'The authored evaluation is six angles $\pm15,\pm30,\pm60$ degrees, all other initial states zero, for 14 physical seconds. Angles are not wrapped. Balanced uses the last 3 seconds and physical angular velocity; it does not itself check falls during the transient. Each new controller reseeds CEM with the configured seed (234 in the authored run).'),
('R25','REPRODUCIBILITY',r'The local regression and Riccati computation use double precision.',r'A fixed seed does not by itself guarantee CUDA bitwise reproducibility; the runner does not request deterministic algorithms. The reuse-data flag loads saved arrays without validating their stored configuration. The new validation manifest records configuration and seeds without treating old outputs as interchangeable.'),
('R26','PLANT / UNIT OMISSION',r'The simulator uses a held-input RK4 update',r'The implemented pendulum is a uniform rod with center of mass $l/2$ and pivot inertia $m_p l^2/3$, rather than a point mass. The hard-coded normalized step uses $1/\sqrt{9.81}$; its stated dimensional formula assumes $l=1$. A length-mismatch experiment must keep the nominal measurement coordinates fixed and change the physical plant, rather than silently renormalizing the learned inputs.'),
('R27','MANIFOLD TERMINOLOGY',r'The learned trajectory set is $\widehat{\mathcal B}_{N,\vartheta}^+=\operatorname{im}(\widehat\Phi_{N,\vartheta})$',r'The image is a smooth graph on an open coordinate domain; a closed input box gives a manifold with corners. The dimension 34 at $N=30$ follows from retaining the coordinates, even for an inaccurate network. It is not an empirical dimension estimate of measured trajectories or a certificate that the learned graph equals the true behavior.'),
]
for ident,category,anchor,annotation in extra:
    notes.append(dict(id=ident,category=category,anchor=anchor,annotation=annotation))
intro='\n\\review{\\textbf{Validation review, October 9, 2026.} This separate annotated transcription preserves source prose and equations. Comments identify notation, missing specification details and limitations. Experiments and the paper audit are linked from validation/results_v1/RESULTS.md.}\n'
rendered=original.replace('\\maketitle','\\maketitle'+intro,1)
blocks=[intro]
for note in notes:
    anchor=note['anchor']
    if rendered.count(anchor)!=1:raise RuntimeError('Nonunique anchor '+note['id'])
    block='\n\\review{\\textbf{'+note['id']+' ('+note['category']+').} '+note['annotation']+'}\n'
    rendered=rendered.replace(anchor,block+anchor if anchor.startswith('\\begin{thebibliography}') else anchor+block,1)
    blocks.append(block)
restored=rendered
for block in reversed(blocks):restored=restored.replace(block,'',1)
assert restored==original,'Source prose was changed'
(out/'trajectory_manifold_implementation_reviewed_v3.tex').write_text(rendered)
(out/'annotations.json').write_text(json.dumps(notes,indent=2)+'\n')
(out/'README.md').write_text('The 27 inline comments preserve the parent-directory source transcription exactly when removed. No labmate source files were edited. PDF compilation is not required for the comments to be read in the TeX file. R22 corrects the handoff precision shorthand: the float32 Q matrix promotes to float64 with simulated NumPy states. The source equations match the implementation; these comments distinguish omissions from false equations.\n')
print(f'{len(notes)} annotations; source text preservation verified')
