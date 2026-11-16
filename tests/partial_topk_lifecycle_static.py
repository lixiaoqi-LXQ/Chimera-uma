#!/usr/bin/env python3
"""CPU control-flow fixture for the public QA-disable path; no CUDA calls.

Evaluate the actual source guards, rather than reimplementing their policy.
Mock reservation generations stand for ownership changes, not GPU allocations.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'Chimera/cpp/src/partial_topk.cu'
BASE = 'd26f01b20115eea4ee81ce4ff2afe1e8bc5fa025'


def condition(source, anchor):
    start = source.index(anchor)
    begin = source.index('(', start)
    depth = 0
    for end in range(begin, len(source)):
        depth += (source[end] == '(') - (source[end] == ')')
        if depth == 0:
            value = source[begin + 1:end]
            value = value.replace('&&', ' and ').replace('||', ' or ')
            value = re.sub(r'!(?!=)', ' not ', value).replace('size_t(', 'int(')
            return ' '.join(value.split())
    raise AssertionError('Unclosed source condition')


class Reservation:
    def __init__(self, source, verification):
        self.reuse = condition(source, 'if (row_capacity >= rows')
        self.bad_input = condition(source, 'if (!scores || !original_ids')
        self.row_capacity = self.selected_capacity = 0
        self.verification_storage = False
        self.generations = 0
        self.handles = None
        self.reserve(65536, 800, verification)

    def reserve(self, rows, selected, verification):
        values = dict(vars(self), rows=rows, selected=selected, verification=verification)
        if eval(self.reuse, {'__builtins__': {}, 'int': int}, values):
            return
        self.row_capacity, self.selected_capacity = rows, selected
        self.verification_storage = verification
        self.generations += 1
        self.handles = object()

    def select(self, verification):
        # A capacity-fit real query follows ensure_candidate_capacity's early
        # return. No selector reserve happens between disable and this guard.
        values = dict(vars(self), scores=True, original_ids=True,
                      rows=30489, count=800, verification=verification)
        if eval(self.bad_input, {'__builtins__': {}, 'int': int}, values):
            raise ValueError('workspace/input extent mismatch')


def main():
    fixed = (ROOT / SOURCE).read_text()
    original = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', BASE + ':' + SOURCE], text=True)
    api = (ROOT / 'Chimera/cpp/include/chimera/chimera_index.cuh').read_text()
    public_disable = api.split('void disable_diagnostic_candidate_verification()', 1)[1].split('}', 1)[0]
    assert 'options_.verify_full_bit = false;' in public_disable
    assert 'worker->options_.verify_full_bit = false;' in public_disable
    assert 'verification ? oracle_input_rows : nullptr' in fixed
    assert 'verification ? selected_rows : nullptr' in fixed
    assert fixed.index('if (!verification) return;') < fixed.index('std::vector<int> actual_ids')

    old = Reservation(original, True)
    old.select(True)
    try:
        old.select(False)  # public disable changed the flag, not the capacity
    except ValueError:
        reproduced = True
    else:
        raise AssertionError('Frozen B08 public-disable failure did not reproduce')

    # Owner plus three batch lanes all receive the public disable operation.
    lanes = [Reservation(fixed, True) for _ in range(4)]
    handles = [lane.handles for lane in lanes]
    assert len({id(handle) for handle in handles}) == 4
    for lane, handle in zip(lanes, handles):
        lane.select(True)
        for _ in range(2):
            lane.select(False)
            lane.reserve(65536, 800, False)
            assert lane.generations == 1 and lane.handles is handle

    no_qa = Reservation(fixed, False)
    try:
        no_qa.select(True)
    except ValueError:
        pass
    else:
        raise AssertionError('Verification accepted missing QA storage')
    generation, handle = no_qa.generations, no_qa.handles
    no_qa.reserve(65536, 800, True)
    assert no_qa.generations == generation + 1 and no_qa.handles is not handle
    no_qa.select(True)
    print(json.dumps(dict(passed=True,CPU_only=True,GPU_executed=False,
        frozen_B08_public_disable_failure_reproduced=reproduced,
        four_lane_true_false_false_reuses_same_owner=True,
        verification_requires_QA_storage=True,false_true_reserve_allocates=True)))


if __name__ == '__main__':
    main()
