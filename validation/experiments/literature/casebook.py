"""Shared fixed nominal cases, independent of any method implementation/imports."""
import json
from common import ROOT


def cases():
    original=json.loads((ROOT/'validation/results_v1/initial_conditions.json').read_text())
    rows=[dict(set='A',index=i,state=x,seconds=14.) for i,x in enumerate(original['sets']['A'])]
    rows += [dict(set='C',index=i,state=original['sets']['C'][i],seconds=14.) for i in original['C_subset']]
    rows += [dict(set='A',index=i,state=original['sets']['A'][i],seconds=60.) for i in (0,2,5)]
    assert len(rows)==59
    return rows
