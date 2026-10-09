"""Matched-input model errors and feasible-alternative optimizer witnesses."""
from campaign import *
CACHE={}


def objective(X,U):
    # Exactly the submitted cost: stages x_0..x_(N-1), terminal 12Q,
    # no difference penalty against the previously applied input.
    return float(np.einsum('bi,ij,bj->',X[:-1],Q,X[:-1])+np.einsum('bi,ij,bj->',U,R,U)
        +12*X[-1]@Q@X[-1]+.002*np.sum(np.diff(U,axis=0)**2))


def replay(state,inputs):
    X=[np.asarray(state,dtype=float).copy()];dt=runner.make_config('dev').dt
    for u in inputs:X.append(runner._plant_step(X[-1],u,dt))
    return np.asarray(X)


def local_candidate(state,gain,horizon):
    X=[np.asarray(state,dtype=float).copy()];U=[];dt=runner.make_config('dev').dt
    for _ in range(horizon):
        u=np.clip(-gain@X[-1],-10.,10.).reshape(1);U.append(u)
        X.append(runner._plant_step(X[-1],u,dt))
    return np.asarray(X),np.asarray(U)


def model_and_gain(seed,arm):
    key=(seed,arm=='Hstar')
    if key not in CACHE:
        path=OUT/'models'/f'episodes3500_seed{seed}'
        model,_=runner.load_transition(path/'transition.pt','cuda')
        gain=np.asarray(json.loads((path/'diagnostics.json').read_text())['local']['gain'])
        if arm=='Hstar':
            from models import OracleTransition
            oracle=OracleTransition('cuda');oracle.xu_std=model.xu_std.detach().clone();model=oracle
        CACHE[key]=(CanonicalBehaviorDecoder(model,30).to('cuda').eval(),gain)
    return CACHE[key]


def witness(seed,arm,state,inputs,prediction,reported_loss):
    decoder,gain=model_and_gain(seed,arm)
    available=inputs is not None
    horizon=len(inputs) if available else 30
    true_plan=replay(state,inputs) if available else None
    true_local,local_inputs=local_candidate(state,gain,horizon)
    with torch.no_grad():
        local_prediction=decoder.rollout(torch.as_tensor(state,dtype=torch.float32,device='cuda'),torch.as_tensor(local_inputs,dtype=torch.float32,device='cuda')).cpu().numpy()
    def number(value):return float(value) if np.isfinite(value) else None
    costs=dict(chosen_model_cost=number(objective(prediction,inputs)) if available else number(reported_loss),
        chosen_true_cost=number(objective(true_plan,inputs)) if available else None,
        local_candidate_model_cost=number(objective(local_prediction,local_inputs)),local_candidate_true_cost=number(objective(true_local,local_inputs)),
        recorded_cem_loss=number(reported_loss),chosen_plan_inputs_available=available,
        local_candidate_inputs_bounded=bool(np.isfinite(local_inputs).all() and np.max(np.abs(local_inputs))<=10.))
    costs.update(compare_costs(costs))
    costs['planner_transition']='true RK4 in float32' if arm=='Hstar' else 'learned transition, CUDA float32'
    costs['interpretation']='A lower feasible internal-model cost witnesses optimizer suboptimality; reversed internal/true preference witnesses decision-relevant model error. Neither proves the sole cause of closed-loop failure.'
    return costs


def compare_costs(costs):
    cm=costs['chosen_model_cost'];ct=costs['chosen_true_cost']
    lm=costs['local_candidate_model_cost'];lt=costs['local_candidate_true_cost']
    result=dict(feasible_better_model_cost=None,model_cost_gap_over_feasible_alternative=None,model_preference_reversal=None)
    if costs['local_candidate_inputs_bounded'] and cm is not None and lm is not None:
        tolerance=1e-5*max(1.,abs(cm),abs(lm))
        result.update(feasible_better_model_cost=bool(lm<cm-tolerance),model_cost_gap_over_feasible_alternative=cm-lm,
            comparison_tolerance=tolerance)
        if ct is not None and lt is not None:
            true_tolerance=1e-5*max(1.,abs(ct),abs(lt))
            result.update(model_preference_reversal=bool(cm<=lm+tolerance and lt<ct-true_tolerance),true_cost_tolerance=true_tolerance)
    return result


def select_representative_failures(rows):
    """Five illustrative 14 s failures, prioritized by matched-arm contrasts."""
    rows=[row for row in rows if row['seconds']==14.]
    lookup={(r['seed'],r['ic_set'],r['ic_index'],r['arm']):r for r in rows}
    failures=[r for r in rows if not r['primary_success'] and r['arm']!='L']
    chosen=[];used=set()
    def succeeds(row,arm):
        other=lookup.get((row['seed'],row['ic_set'],row['ic_index'],arm))
        return other is not None and other['primary_success']
    def pick(predicate,reason,angle=60.):
        candidates=[r for r in failures if r['case'] not in used and predicate(r)]
        if not candidates:return
        candidates.sort(key=lambda r:(r['seed'],r['ic_set']!='B',abs(abs(np.rad2deg(r['initial_state'][2]))-angle),r['case']))
        row=candidates[0];chosen.append(dict(row,diagnostic_selection=reason));used.add(row['case'])
    pick(lambda r:r['arm']=='H' and succeeds(r,'Hstar'),'H failed where matched H* succeeded')
    if not chosen:pick(lambda r:r['arm']=='H' and succeeds(r,'L'),'H failed where matched L succeeded')
    before=len(chosen)
    pick(lambda r:r['arm']=='H' and r['fell_within_1s_of_switch_exit'],'Fall within 1 s of leaving local feedback')
    if len(chosen)==before:
        pick(lambda r:r['arm']=='H' and succeeds(r,'L') and r['switch_toggles']>0,'H missed with switch toggles; matched L succeeded')
    pick(lambda r:r['arm']=='M1' and r['no_fall'] and succeeds(r,'L'),'M1 missed settling without a fall; L succeeded',15.)
    pick(lambda r:r['arm']=='M' and r['no_fall'] and succeeds(r,'L'),'Queued M missed settling without a fall; L succeeded',15.)
    pick(lambda r:r['arm']=='Hstar' and succeeds(r,'L'),'True-model H* failed where matched L succeeded')
    for arm in ('H','M1','M','Hstar'):
        if len(chosen)>=5:break
        pick(lambda r,arm=arm:r['arm']==arm,'Additional illustrative failure; contrast unavailable')
    while len(chosen)<min(5,len(failures)):
        before=len(chosen);pick(lambda r:True,'Additional illustrative failure; contrast unavailable')
        if len(chosen)==before:break
    return chosen[:5]
