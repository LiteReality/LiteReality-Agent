"""Bounded, evidence-backed repair against a persistent scan baseline."""

import copy
import math
from pathlib import Path

from .adjust import Violation, check


def fidelity_errors(shell, baseline):
    originals = dict(baseline['objects'])
    for obj in baseline['objects'].values():
        originals.update(obj.get('merged_members', {}))
    errors = []
    for oid, obj in shell['objects'].items():
        old = originals.get(oid)
        if old is None:
            errors.append(oid)
            continue
        max_scale = 0.15 if obj.get('category') in {'table', 'desk', 'chair', 'sofa', 'bed'} else 0.35
        if (any(not math.isfinite(v) for v in obj['center'])
                or math.dist(obj['center'][:2], old['center'][:2]) > 0.6
                or abs(obj['center'][2] - old['center'][2]) > 0.08
                or any(not math.isfinite(v) or abs(v - w) > max_scale * w
                       for v, w in zip(obj['size'], old['size']))):
            errors.append(oid)
    for oid, obj in baseline['objects'].items():
        if oid not in shell['objects'] and not (
                obj.get('merged_members') and set(obj['merged_members']) <= shell['objects'].keys()):
            errors.append(oid)
    return sorted(set(errors))


def quality(shell):
    errors = [v for v in check(shell) if v.severity == 'error']
    return len(errors), round(sum(v.magnitude or 0 for v in errors), 4)


def solve(shell, baseline, *, use_agent, max_rounds=6):
    from .agent import apply_proposals, propose
    from .repair import repair

    current, moves, log = repair(shell)
    invalid = fidelity_errors(current, baseline)
    if invalid:
        # A legal arrangement can still be an implausible distortion. Keep the measured scene
        # available to the agent instead of treating it as a successful repair.
        current = copy.deepcopy(shell)
        moves = []
        log.append({'action': 'fidelity_rejected', 'objects': invalid})
    root = Path(shell.get('meta', {}).get('batch_dir', '.'))
    current.setdefault('meta', {})['scan_baseline'] = copy.deepcopy(baseline['objects'])
    feedback = current['meta'].setdefault('layout_feedback', {})
    stop = 'agent_disabled'
    calls = 0
    stalled_rounds = 0
    for round_id in range(max_rounds if use_agent else 0):
        errors = [v for v in check(current) if v.severity == 'error']
        invalid = fidelity_errors(current, baseline)
        if not errors and not invalid:
            stop = 'validated'
            break
        targets = {v.object: v for v in errors}
        for v in errors:
            if v.other in current['objects']:
                targets.setdefault(v.other, Violation(v.other, v.kind, v.detail, v.magnitude, v.object))
        for oid in invalid:
            targets.setdefault(oid, Violation(oid, 'fidelity', 'excessive change from scan baseline'))
        changed = False
        for oid, violation in targets.items():
            # An accepted edit plus deterministic repair may resolve several original targets.
            # Recheck before paying for another review of a collision that no longer exists.
            live_errors = [v for v in check(current) if v.severity == 'error']
            invalid = fidelity_errors(current, baseline)
            if not live_errors and not invalid:
                break
            relevant = [v for v in live_errors if oid in (v.object, v.other)]
            if not relevant and oid not in invalid:
                continue
            if relevant:
                latest = relevant[0]
                violation = (latest if latest.object == oid else Violation(
                    oid, latest.kind, latest.detail, latest.magnitude, latest.object))
            if oid not in current['objects'] or not list((root / 'references' / oid).glob('rank*.jpg')):
                log.append({'action': 'agent_skipped', 'object': oid, 'reason': 'no reference images'})
                continue
            calls += 1
            proposal = propose(current, violation, root, suspicion='remaining geometry/fidelity violation')
            if not proposal:
                continue
            trial, decisions = apply_proposals(current, [proposal])
            feedback.setdefault(oid, []).extend(decisions)
            log.append({'action': 'agent_review', 'round': round_id + 1, 'decisions': decisions})
            if trial['objects'] == current['objects']:
                continue
            candidate, more_moves, more_log = repair(trial)
            if fidelity_errors(candidate, baseline):
                candidate, more_moves, more_log = trial, [], []
            if not fidelity_errors(candidate, baseline) and quality(candidate) < quality(current):
                current = candidate
                current['meta']['layout_feedback'] = feedback
                moves += more_moves
                log += more_log
                changed = True
            else:
                rejected = {'action': 'candidate_rejected', 'object': oid,
                            'fidelity_errors': fidelity_errors(candidate, baseline),
                            'before': quality(current), 'after': quality(candidate)}
                log.append(rejected)
                feedback.setdefault(oid, []).append(rejected)
        if not changed:
            stalled_rounds += 1
            if stalled_rounds >= 2 or calls == 0:
                stop = 'no_progress'
                break
        else:
            stalled_rounds = 0
    else:
        if use_agent:
            stop = 'round_limit'
    if quality(current)[0] == 0 and not fidelity_errors(current, baseline):
        stop = 'validated'
    return current, moves, log, {'agent_calls': calls, 'stop_reason': stop,
                                'fidelity_errors': fidelity_errors(current, baseline)}
