# Graduation Project and Thesis Work Log (Weeks 1–33)

> Compiled based on the project workflow in `README.md` and the thesis draft *Design and Implementation of a Classroom Attention Analysis System Based on Facial Expressions (Second Draft, 5.23 Annotated)*. Weeks 1–24 focus on system development, while Weeks 25–33 focus on thesis writing and revision.

## Weeks 1–24: Graduation Project Development Phase

### Week 1
- Defined the project direction: classroom attention analysis based on facial emotion cues.
- Clarified the overall objective: build a full pipeline of “capture–recognition–scoring–visualization–reporting”.
- Confirmed scope with advisor: use common classroom camera videos without extra hardware.
- Set up weekly meetings and progress logs with explicit deliverables (code, charts, documents).

### Week 2
- Reviewed literature on classroom evaluation, emotion recognition, and attention analysis.
- Selected the initial technical route: YOLO face detection + emotion recognition + mapping-based scoring.
- Compared candidate models and inference workflows, prioritizing low deployment cost.
- Produced a technical pre-study note with feasible options and risk items.

### Week 3
- Completed project structure planning and baseline environment setup.
- Refined module boundaries: frame extraction, labeling, mapping, reporting, and app entry.
- Unified code style and directory naming conventions for reproducibility.
- Built a minimal runnable prototype to verify dependency and script-chain availability.

### Week 4
- Implemented frame extraction with configurable sampling intervals.
- Standardized frame naming and timestamp indexing.
- Designed dataset folder hierarchy (course/date/segment) for batch labeling.
- Generated first classroom frame samples and conducted manual quality checks.

### Week 5
- Integrated face detection inference and bounding-box output.
- Started collecting false-positive and false-negative cases for tuning.
- Added detection-result visualization export to build an issue sample set.
- Summarized performance under backlight, side-face, and occlusion conditions.

### Week 6
- Integrated emotion recognition and connected “face ROI → emotion probabilities”.
- Verified basic usability for multi-face frames.
- Added exception handling for empty/out-of-bound ROIs.
- Produced first per-frame logs with face position, dominant emotion, and confidence.

### Week 7
- Added box-quality filters (minimum size and aspect ratio).
- Reduced interference from low-quality samples in emotion recognition.
- Re-ran sample sets after threshold updates to compare false detections.
- Versioned parameters and created a “parameter–effect” tracking record.

### Week 8
- Completed semi-automatic labeling flow (presence and emotion labels).
- Generated the first face-level labeled dataset.
- Standardized label definitions and conflict-resolution rules.
- Ran small-scale cross-review to improve labeling consistency.

### Week 9
- Added attention-labeling workflow (high/mid/low).
- Linked frame-level and face-level samples.
- Introduced secondary verification to reduce subjective bias.
- Produced annotation guidelines for future dataset expansion.

### Week 10
- Cleaned labeled data: missing values, duplicates, and outliers.
- Built a training-ready dataset for mapping calibration.
- Added data versioning and change logs for traceability.
- Output sample statistics (class distribution, volume, validity ratio).

### Week 11
- Designed a data-driven “emotion → attention” mapping method.
- Produced first mapping parameters and config file.
- Compared heuristic fixed scores vs calibrated mapping for interpretability.
- Recorded effects of mapping changes on trend curves.

### Week 12
- Implemented mapping loading and switching mechanisms.
- Compared outputs from default mapping and calibrated mapping.
- Improved config fault tolerance for missing fields.
- Generated controlled comparison results for thesis method section.

### Week 13
- Implemented frame-level aggregation (multi-person averaging).
- Introduced EMA smoothing to reduce curve jitter.
- Compared smoothing factors for stability–responsiveness trade-off.
- Fixed recommended parameter ranges in experiment notes.

### Week 14
- Integrated SQLite persistence for frame-level summaries and face-level events.
- Completed traceable storage of analysis outputs.
- Designed key indexes to improve offline query efficiency.
- Added retry and logging for database write failures.

### Week 15
- Built offline report-generation framework.
- Exported summary and trend CSV files.
- Defined reporting metrics and thresholds.
- Standardized report-output folder and naming rules.

### Week 16
- Added chart exports (trend/distribution) and report text output.
- Completed initial “online analysis + offline review” loop.
- Improved chart style and annotations for readability.
- Prepared first chart set usable in thesis writing.

### Week 17
- Optimized UI layout and runtime status prompts.
- Improved long-run stability details.
- Enhanced thread-state observability.
- Fixed UI lag and refresh-delay issues.

### Week 18
- Improved dual input flow (video file and camera).
- Fixed read-failure and end-of-stream handling.
- Added fallback behavior for uninterrupted demos.
- Supplemented test cases for source switching.

### Week 19
- Tuned detection parameters (confidence, IoU, etc.).
- Balanced inference speed and recognition quality.
- Compared performance across resolutions and set recommended specs.
- Summarized tuning findings for thesis experiments.

### Week 20
- Improved emotion-recognition exception handling and filtering.
- Optimized model loading and runtime feedback.
- Added low-confidence suppression to reduce short-term noise.
- Improved startup-stage prompts and usability.

### Week 21
- Performed end-to-end integration: detection, recognition, scoring, storage, plotting.
- Fixed key field mapping and data-type issues.
- Ran stress tests and optimized bottleneck paths.
- Closed integration issues through tracked problem lists.

### Week 22
- Added linked trend-chart display capability.
- Optimized dynamic refresh and visual interaction.
- Improved front-end adaptive layout and timeline display.
- Polished demo UI details.

### Week 23
- Completed start/stop controls and interruptible workflow design.
- Improved classroom-demo operability.
- Added button-state protection against misoperations.
- Rehearsed demo scripts and fixed discovered issues.

### Week 24
- Integrated one-click “generate report/open report folder”.
- Completed feature closing for the graduation project.
- Organized diagrams, screenshots, and key data tables for thesis use.
- Consolidated limitations and future improvements for discussion chapter input.

## Weeks 25–33: Thesis-Focused Phase (Key Stage)

### Week 25 (Thesis Preparation and Material Archiving)
- Organized thesis materials from system outputs: architecture diagrams, process diagrams, module notes, and test screenshots.
- Mapped development outcomes to chapter structure: Introduction–Method–Implementation–Testing–Conclusion.
- Finalized writing checklist: abstract, keywords, requirement analysis, key methods, and result analysis.

### Week 26 (Framework and Draft Chapters)
- Built the full chapter structure from the thesis table of contents.
- Expanded background, significance, and related work to emphasize classroom deployment pain points.
- Clarified innovation claims: deployability, explainable scoring, and complete engineering loop.

### Week 27 (Focus on Chapters 2 & 3)
- Completed theory chapter drafts: YOLO, emotion recognition, and attention concepts.
- Completed requirement analysis across six modules.
- Completed overall architecture description: data acquisition, intelligent analysis, storage, and presentation layers.

### Week 28 (Focus on Chapter 4: Key Methods)
- Detailed face-detection method design: thresholds, NMS, size/aspect-ratio filtering.
- Completed scoring formulation: emotion-probability weighting, frame-level aggregation, temporal smoothing.
- Added explainability argument: mapping calibrated by labeled data instead of purely subjective assignment.

### Week 29 (Focus on Chapter 5: System Implementation)
- Completed environment and tech-stack descriptions.
- Completed storage and offline-report implementation sections.
- Unified alignment between implementation details, interface screenshots, and output artifacts.

### Week 30 (Focus on Chapter 6: Testing and Analysis)
- Finalized testing goals, environment, data source, and evaluation metrics.
- Summarized functional test outcomes and system stability conclusions.
- Wrote analysis on attention distribution, trend dynamics, and low-attention segment value.

### Week 31 (Second Draft Refinement)
- Produced the second full draft and improved chapter transitions.
- Fixed structural issues such as numbering, terminology consistency, and logic jumps.
- Completed chart-text consistency checks.

### Week 32 (Advisor-Comment Revision)
- Revised by annotated comments: sharpened background statements, normalized method descriptions, and constrained conclusion boundaries.
- Corrected format details: heading hierarchy, figure captions, references, and typos.
- Added objective discussion on deployment feasibility and limitations.

### Week 33 (Finalization and Defense Preparation)
- Completed final proofreading and formatting checks.
- Consolidated defense materials: problem statement, architecture, core methods, results, innovation points, and limitations.
- Finalized defense script and presentation flow to close the project.
