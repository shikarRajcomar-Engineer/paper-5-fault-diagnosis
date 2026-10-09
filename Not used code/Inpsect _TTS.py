"""
TTS_check_truth.py — IS THE GROUND TRUTH RIGHT?
===============================================
Read-only. Writes nothing. Two questions.

QUESTION 1 — WHY DOES fault01 LOOK EMPTY?
    The earlier check compared the mean before onset with the mean after.
    Fault 1 is a sinusoid, 6*sin((t-200)/6pi), which has zero mean, so it
    cancels out completely under that test. Measuring how much each sensor
    VARIES, rather than where it sits, makes an oscillation visible.

QUESTION 2 — WHICH SENSOR DOES EACH SENSOR-FAULT SIT ON?
    The repo README lists faults 4, 5 and 6 as "Sensor" faults but never
    says which sensor each one is attached to. The current ground truth
    assumes 4 -> h1, 5 -> h2, 6 -> h3, which was a guess.

    Each of those faults is a known ramp. A sensor fault adds that ramp
    directly to one measurement, so the affected sensor should shift by an
    amount we can predict from the README equation. Comparing the
    predicted shift against each sensor's actual shift identifies which
    one carries the fault.

FAULT EQUATIONS (from the repo README)
    f4 = -0.005 (t - 200)     Sensor
    f5 =  0.005 (t - 200)     Sensor
    f6 =  0.008 (t - 200)     Sensor

RUN IT
    python TTS_check_truth.py
"""

import argparse
import os

import numpy as np


# ----------------------------------------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------------------------------------

SENSOR_NAMES = ['h1', 'h2', 'h3', 'Q1', 'Q2']
KEEP_COLS = [0, 1, 2, 3, 4]        # the designed files hold exactly these
FAULT_ONSET = 200

# Ramp slopes for the three sensor faults, from the README.
SENSOR_FAULT_SLOPES = {'fault04': -0.005, 'fault05': 0.005, 'fault06': 0.008}

# The ground truth currently in use, for comparison.
CURRENT_TRUTH = {
    'fault01': ['Q1', 'Q2'], 'fault02': ['Q1'], 'fault03': ['Q2'],
    'fault04': ['h1'], 'fault05': ['h2'], 'fault06': ['h3'],
    'fault07': ['h1'], 'fault08': ['h2'], 'fault09': ['h3'],
    'fault10': ['h1', 'h3'], 'fault11': ['h2'], 'fault12': ['h2', 'h3'],
}


def load_faults(data_dir):
    """Load every fault array from the designed test file."""
    from scipy.io import loadmat

    path = os.path.join(data_dir, '[designed_test].mat')
    if not os.path.exists(path):
        raise SystemExit(f'Not found: {path}')
    raw = loadmat(path)
    faults = {k: np.asarray(v).astype(float)
              for k, v in raw.items() if k.startswith('fault')}
    print(f'  loaded {len(faults)} fault array(s) from '
          f'{os.path.basename(path)}')
    return faults


# ----------------------------------------------------------------------------------------------------------
# 1. Question 1 — variability, which sees oscillations
# ----------------------------------------------------------------------------------------------------------

def variability_report(faults):
    """
    How much MORE each sensor varies after onset than before.

    A ramp raises the mean; an oscillation does not, but it does raise the
    variability. Reporting the ratio of post-onset to pre-onset standard
    deviation catches both.
    """
    print('\n' + '-' * 74)
    print('QUESTION 1 — VARIABILITY AFTER ONSET (catches oscillating faults)')
    print('-' * 74)
    print(f'\n  {"fault":<10}' + ''.join(f'{n:>9}' for n in SENSOR_NAMES)
          + '   biggest')

    for key in sorted(faults):
        array = faults[key]
        before, after = array[:FAULT_ONSET], array[FAULT_ONSET:]
        ratios = []
        for column in KEEP_COLS:
            pre = before[:, column].std()
            post = after[:, column].std()
            ratios.append(post / pre if pre > 1e-9 else 0.0)
        biggest = SENSOR_NAMES[int(np.argmax(ratios))]
        print(f'  {key:<10}' + ''.join(f'{r:>9.2f}' for r in ratios)
              + f'   {biggest}')
    print('\n  A ratio near 1.0 means that sensor is as steady as it was.')
    print('  fault01 should now show a clear response, unlike the mean test.')


# ----------------------------------------------------------------------------------------------------------
# 2. Question 2 — which sensor each ramp is attached to
# ----------------------------------------------------------------------------------------------------------

def predicted_shift(slope, n_after):
    """
    Average value of a ramp over the post-onset window.

    The fault is slope * (t - 200), so at sample i after onset its value is
    slope * i. Averaged over the window that is slope * (n - 1) / 2.
    """
    return abs(slope) * (n_after - 1) / 2.0


def sensor_fault_report(faults):
    """
    Match each sensor-fault's predicted shift against the observed shifts.

    A sensor fault is added straight onto one measurement, so the sensor it
    sits on should shift by very close to the predicted amount, in the
    sensor's own physical units. Other sensors move only through process
    coupling, which gives a different and usually smaller shift.
    """
    print('\n' + '-' * 74)
    print('QUESTION 2 — WHICH SENSOR DOES EACH SENSOR-FAULT SIT ON?')
    print('-' * 74)

    for key, slope in SENSOR_FAULT_SLOPES.items():
        if key not in faults:
            print(f'\n  {key}: not present')
            continue
        array = faults[key]
        before, after = array[:FAULT_ONSET], array[FAULT_ONSET:]
        expected = predicted_shift(slope, len(after))

        print(f'\n  {key}  (fault = {slope} * (t-200), '
              f'so the affected sensor should shift by about '
              f'{expected:.2f} units)')
        print(f'    {"sensor":<8} {"actual shift":>14} {"vs predicted":>14}'
              f'   reading')

        best_name, best_gap = None, np.inf
        for index, column in enumerate(KEEP_COLS):
            name = SENSOR_NAMES[index]
            observed = abs(after[:, column].mean() - before[:, column].mean())
            gap = abs(observed - expected) / (expected + 1e-9)
            if gap < best_gap:
                best_name, best_gap = name, gap
            reading = ('MATCHES the fault equation' if gap < 0.25
                       else 'close' if gap < 0.60 else '')
            print(f'    {name:<8} {observed:>14.3f} {100 * gap:>13.0f}%'
                  f'   {reading}')

        stated = CURRENT_TRUTH.get(key, ['?'])[0]
        if best_name == stated:
            print(f'    -> best match {best_name}, ground truth says '
                  f'{stated}.  AGREES')
        else:
            print(f'    -> best match {best_name}, but ground truth says '
                  f'{stated}.  DISAGREES')


def summarise(faults):
    """State what, if anything, needs changing in the ground truth."""
    print('\n' + '=' * 74)
    print('WHAT THIS MEANS')
    print('=' * 74)

    disagreements = []
    for key, slope in SENSOR_FAULT_SLOPES.items():
        if key not in faults:
            continue
        array = faults[key]
        before, after = array[:FAULT_ONSET], array[FAULT_ONSET:]
        expected = predicted_shift(slope, len(after))
        best_name, best_gap = None, np.inf
        for index, column in enumerate(KEEP_COLS):
            observed = abs(after[:, column].mean() - before[:, column].mean())
            gap = abs(observed - expected) / (expected + 1e-9)
            if gap < best_gap:
                best_name, best_gap = SENSOR_NAMES[index], gap
        stated = CURRENT_TRUTH.get(key, ['?'])[0]
        if best_name != stated:
            disagreements.append((key, stated, best_name))

    if not disagreements:
        print('\n  The sensor-fault assignments in the ground truth agree')
        print('  with the fault equations. Nothing to change.')
    else:
        print('\n  These entries disagree with the fault equations:')
        for key, stated, best in disagreements:
            print(f'    {key}: ground truth says {stated}, '
                  f'the data says {best}')
        print('\n  Attribution accuracy is scored against this table, so a')
        print('  wrong entry counts a correct answer as wrong. Worth fixing')
        print('  before any results are generated.')


def main():
    parser = argparse.ArgumentParser(
        description='Check the three-tank ground truth against the data.')
    parser.add_argument('--data-dir', default='TTS_Dataset')
    args = parser.parse_args()

    print('=' * 74)
    print('THREE-TANK GROUND TRUTH CHECK')
    print('=' * 74)

    faults = load_faults(args.data_dir)
    variability_report(faults)
    sensor_fault_report(faults)
    summarise(faults)

    print('\nNothing was modified. Paste this output back.')


if __name__ == '__main__':
    main()