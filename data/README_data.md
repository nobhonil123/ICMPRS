# ICMPRS Data Dictionary

## Overview

Three CSV files, one per modality, sharing consistent `subject_id` and `label`
columns. Total: **62 feature dimensions** across 1,995 synthetic participants.

---

## Common Columns (all files)

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | Unique ID (`ICMPRS_0001` to `ICMPRS_1995`) |
| `label` | int | `0` = Healthy Control, `1` = Parkinson's Disease |
| `hy_stage` | int | Hoehn & Yahr stage (`0` = HC, `1`/`2`/`3` = PD severity) |
| `sex` | str | `M` or `F` |
| `dialect` | str | North-Hindi / West-Marathi / South-Tamil / East-Bengali |
| `device` | str | Budget Android / Mid-Range Android / Feature Phone |

---

## voice_final.csv (25 feature columns)

### Standard Acoustic Features (20)

| Column | Unit | Description | Source |
|--------|------|-------------|--------|
| `jitter_local` | ratio | Cycle-to-cycle pitch perturbation | Little 2009 |
| `shimmer_local` | ratio | Cycle-to-cycle amplitude perturbation | Tsanas 2012 |
| `hnr_db` | dB | Harmonics-to-noise ratio | Harel 2004 |
| `pitch_mean_hz` | Hz | Mean fundamental frequency | Little 2009 |
| `rpde` | — | Recurrence period density entropy | Little 2007 |
| `dfa` | — | Detrended fluctuation analysis exponent | Little 2007 |
| `spread1` | — | Nonlinear pitch spread (measure 1) | Little 2009 |
| `spread2` | — | Nonlinear pitch spread (measure 2) | Little 2009 |
| `mfcc_1` .. `mfcc_5` | — | Mel-frequency cepstral coefficients 1–5 | Tsanas 2012 |
| `jitter_rap` | ratio | Relative average perturbation | Little 2009 |
| `shimmer_apq11` | ratio | 11-point amplitude perturbation quotient | Tsanas 2012 |
| `nhr` | ratio | Noise-to-harmonics ratio | Harel 2004 |
| `pitch_std_hz` | Hz | Pitch standard deviation | Little 2009 |
| `voiced_fraction` | ratio | Fraction of voiced frames | — |
| `speech_rate` | syll/s | Syllables per second | — |
| `pause_ratio` | ratio | Pause time / total time | — |

### Novel Feature: RCA Degradation Index (1)

| Column | Unit | Description |
|--------|------|-------------|
| `rca_index` | Hz | Mean F₂ deviation during Hindi retroflex consonants from healthy Indian norm |

### Novel Feature: Nasalisation Residual Energy — NRE (3)

| Column | Unit | Description |
|--------|------|-------------|
| `nre_nasal_energy_ratio` | ratio | Spectral energy in 250–450 Hz band / total energy during target vowels |
| `nre_velopharyngeal_leakage` | ratio | Nasal-band energy during oral consonants / total energy |
| `nre_nasal_formant_bw_hz` | Hz | Bandwidth of first nasal formant |

---

## hand_final.csv (21 feature columns)

### Standard Handwriting Features (15)

| Column | Unit | Description | Source |
|--------|------|-------------|--------|
| `pen_velocity_cm_s` | cm/s | Mean pen-tip velocity | Drotár 2016 |
| `drawing_time_s` | s | Total task time | Drotár 2016 |
| `pen_pressure_norm` | 0–1 | Normalised pen pressure | Impedovo 2019 |
| `velocity_direction_changes` | count | Number of velocity direction reversals | Drotár 2016 |
| `tremor_frequency_hz` | Hz | Dominant tremor frequency in pen signal | Impedovo 2019 |
| `task_completion_time_s` | s | Time to complete writing task | Drotár 2016 |
| `stroke_length_cm` | cm | Mean stroke length | Impedovo 2019 |
| `stroke_width_mm` | mm | Mean stroke width | — |
| `curvature_mean` | 1/cm | Mean curvature of strokes | — |
| `curvature_std` | 1/cm | Std deviation of curvature | — |
| `writing_area_cm2` | cm² | Bounding box area of writing | — |
| `horizontal_drift_mm` | mm | Horizontal drift from baseline | — |
| `vertical_drift_mm` | mm | Vertical drift from baseline | — |
| `acceleration_mean_cm_s2` | cm/s² | Mean pen acceleration | — |
| `deceleration_peaks` | count | Number of deceleration peaks | — |

### Novel Feature: Devanagari Pen-Lift Dynamics — DPLD (3)

| Column | Unit | Description |
|--------|------|-------------|
| `dpld_lift_count` | count | Number of pen-lifts during Devanagari writing |
| `dpld_lift_duration_ms` | ms | Mean duration of pen-lift events |
| `dpld_restart_velocity_cm_s` | cm/s | Mean pen velocity after lift-restart |

### Novel Feature: Akshar Junction Hesitation Index — AJHI (3)

| Column | Unit | Description |
|--------|------|-------------|
| `ajhi_conjunct_transition_ms` | ms | Time navigating junction between conjunct strokes |
| `ajhi_conjunct_pressure_spike` | ratio | Peak pressure at junction / mean adjacent pressure |
| `ajhi_inter_akshar_pause_ratio` | ratio | Pause at conjuncts / pause at simple characters |

---

## gait_final.csv (16 feature columns)

### Standard Gait Features (15)

| Column | Unit | Description | Source |
|--------|------|-------------|--------|
| `stride_length_m` | m | Mean stride length | Wahid 2015 |
| `walking_speed_m_s` | m/s | Mean walking speed | Zeng 2016 |
| `cadence_steps_min` | steps/min | Cadence | Wahid 2015 |
| `cadence_variability` | CV | Coefficient of variation of cadence | Wahid 2015 |
| `step_width_cm` | cm | Lateral step width | — |
| `lateral_trunk_sway_deg` | ° | Lateral trunk sway angle | — |
| `freeze_index` | 0–1 | Freezing-of-gait index | — |
| `stride_time_s` | s | Mean stride time | — |
| `swing_phase_pct` | % | Swing phase percentage | — |
| `stance_phase_pct` | % | Stance phase percentage | — |
| `double_support_pct` | % | Double-support phase percentage | — |
| `stride_asymmetry` | ratio | Left-right stride asymmetry | — |
| `step_time_variability_ms` | ms | Step-to-step time variability | — |
| `toe_clearance_cm` | cm | Minimum toe clearance | — |
| `arm_swing_asymmetry` | ratio | Left-right arm swing asymmetry | — |

### Novel Feature: Barefoot Stride Variance — BSV (1)

| Column | Unit | Description |
|--------|------|-------------|
| `bsv_normalised` | ratio | Surface-normalised stride variability (neurological component) |

### Additional Column

| Column | Type | Description |
|--------|------|-------------|
| `footwear` | str | Barefoot (paved) / Barefoot (unpaved) / Chappals / Shoes |

---

## Simulation Parameters

All features drawn from truncated normal or Poisson distributions calibrated
to published clinical norms. See `simulator/generate_icmprs.py` for exact
parameter values and citations.

| Feature | PD Mean | HC Mean | Source |
|---------|---------|---------|--------|
| Jitter (local) | 0.028 | 0.005 | Little et al. 2009 |
| Shimmer (local) | 0.174 | 0.048 | Tsanas et al. 2012 |
| HNR (dB) | 13.6 | 22.9 | Harel et al. 2004 |
| Stride (m) | 0.68 | 1.09 | Wahid et al. 2015 |
| Walk speed (m/s) | 0.62 | 1.10 | Zeng et al. 2016 |
| RCA (Hz) | 168 | 38 | Godara & Das 2020 |
| NRE: NER | 0.41 | 0.22 | Ohala 1975 / Logemann 1978 |
| DPLD lift dur. (ms) | 724 | 210 | Drotár 2016 / Bharati 1995 |
| AJHI trans. (ms) | 412 | 185 | Vaid & Gupta 2002 / Kao 2010 |
| BSV | 0.32 | 0.08 | Hollman et al. 2011 |
