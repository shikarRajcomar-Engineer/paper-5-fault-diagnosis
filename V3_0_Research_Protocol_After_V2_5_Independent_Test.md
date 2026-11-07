# V2.5 independent TEP validation — frozen results and V3.0 development protocol

## Status

This document records existing results and a proposed next experiment. It is not a change to V2.5 or a claim that V3.0 has been validated.

## Dataset and frozen method

- 52 separate single-channel autoencoders and MinMaxScaler fitted on fault-free TEP training data.
- Absolute scaled reconstruction error per channel.
- V2.5: q99 per-channel normal-training reconstruction-error thresholds; first onset of three consecutive exceedances; ascending onset, with threshold-normalized strength as tie-breaker.
- Root-cause labels used only after ranking for evaluation.
- Historical fault-training sequences: 500 samples/run with intervention at sample 21.
- Independent fault-testing sequences: 960 samples/run with intervention at sample 161.
- Testing evaluation: 15 prespecified origin-labeled fault mechanisms, 500 runs each (7,500 runs). Faults 16–20 excluded from origin-accuracy denominator because their cause mapping is not established.

## Reported accuracy (do not revise retroactively)

| Metric | Historical fault training | Independent full testing | Post-hoc matched 500-sample testing |
|---|---:|---:|---:|
| Top-1 (%) | 20.25 | 3.80 | 17.37 |
| Top-3 (%) | 39.63 | 25.37 | 37.44 |
| Top-5 (%) | 41.15 | 37.16 | 40.51 |
| Median true-origin rank | 12 | 13 | 13 |

The **primary independent test result is 25.37% Top-3**. The matched-window result uses known intervention timing and **must never be presented as label-free deployment or as the primary independent V2.5 test score**. Its fixed window is samples 141–640, making injection at relative sample 21.

The matched-window diagnostic recovered 12.07 of the 14.26 percentage points lost in Top-3 compared with the historical training result (~84.6%), but this is sensitivity evidence, not a causal decomposition of the loss.

## Pre-fault onset audit (full 960 samples)

- 83.88% of run-winning sensors triggered before the known injection time.
- True origin triggered before injection in 0.533% of runs.
- True origin had some sustained threshold crossing in 73.4% of runs, often too late to rank highly.
- Median number of pre-fault abnormal sensors per run: 1.
- Most frequent pre-injection winners: xmeas_39 (1350), xmeas_41 (1320), xmeas_40 (1185), xmeas_37 (1080), totaling 4935/6291 pre-fault winners (78.4%).
- Persistent difficult origins after matched-window adjustment include faults 1, 2, 8, 12, 13, 15; diagnosis cannot be fixed solely by selecting a shorter time interval.

## Scientific interpretation and limitations

Two observed limitations must be separated:

1. **Episode identification:** V2.5 treats the earliest sustained deviation anywhere in the entire run as evidence of the initiating fault. Long healthy observation intervals create opportunities for unrelated early crossings.
2. **Origin discrimination / observability:** After aligning observation windows, some true origins still rank very poorly, and some origins barely cross their normal thresholds. Earlier downstream responses are not reliable evidence of causation.

The comparison involves different observation durations and available pre-/post-intervention samples; it does not strictly establish that the longer healthy interval *alone* caused the observed accuracy gap. Test data have now been inspected in detail, so future improvements cannot claim a wholly untouched independent confirmation on these same testing runs.

## V3.0 candidate: label-free episode-aware origin attribution

**Hypothesis:** Identifying a *new collective abnormal episode* before ranking sensors will reduce premature false-alarm winners, while explicit uncertainty/abstention can avoid falsely confident root-cause claims when origin signals are weak.

Proposed sequence:

1. From normal TRAINING-only errors, estimate expected per-sensor background exceedances and a process-level change statistic (e.g. changes in number of abnormal channels or innovations relative to a rolling baseline); no fault-origin labels in thresholds.
2. Detect a process-wide change point **without using faultNumber, simulated injection time, or ground-truth source**. A fixed-length pre-change reference is allowed only if it is available at deployment; no future leakage.
3. Rank candidate origins using evidence in a prespecified interval relative to the **detected** change point; the time window may not be centered on an oracle intervention timestamp.
4. If evidence is insufficient, output uncertainty/abstention along with a ranking; report coverage separately from correctness.
5. Preserve process-variable identities and report false-alarm/early-alarm rates on held-out fault-free **training runs**, not only conditional fault-origin accuracy.

**Development protocol:**

- Split training runs by simulationRun, never by individual time samples, for exploratory development versus internal validation.
- Evaluate candidate episode detectors using normal-training-only calibration, frozen sensor AEs, and fault-training runs.
- Document all design decisions and tuned hyperparameters before reuse of independent testing data.
- Evaluate changes to Top-1/3/5, median rank, abstention/coverage, episode latency, and false episode alerts.
- Use injected fault time and known origins solely for post-hoc metrics.
- Treat any future result on the already-examined TEP test set as **test-set-exposed follow-up**, not a new pristine holdout. Strong claims should use a separate fresh benchmark, a withheld subset not previously examined, or a new simulator-generated run set.

## Repository preservation

Do not edit `5.1_V2_5_Leading_Edge_Attribution.py.py`, its earlier output, or the existing normal-training error cache. Diagnostics 6.6 and 6.7 remain as read-only evidence; V3.0 code must live in separate new scripts and directories.
