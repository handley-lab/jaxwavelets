# Not Vibes: How an AI Coding Agent Reproduced a Scientific Library to Machine Precision

**Will Handley, Institute of Astronomy, University of Cambridge**

## Abstract

[PyWavelets](https://pywavelets.readthedocs.io/) encodes two decades of hard-won numerical decisions: boundary conventions, filter alignments, and floating-point behaviours discovered through real scientific use. I wanted that *embodied intelligence* inside [JAX](https://github.com/google/jax), without losing numerical fidelity. In one AI-assisted [Claude Code](https://www.anthropic.com/claude-code) session, the agent produced [jaxwavelets](https://github.com/handley-lab/jaxwavelets) — a JAX-native implementation matching PyWavelets to `atol=1e-14` across 1,177 tests. I wrote no code, but made the scientific decisions: tolerances, API design, GPU arithmetic choices, scope, and review strategy. This case study argues that AI coding agents can produce credible scientific software when the task is **reference-guided translation**, the **oracle is mature**, and **review is adversarial**: 65 external LLM reviews rejected 60% of submissions before merge.

## 1. Embodied Intelligence

A mature scientific library like [PyWavelets](https://pywavelets.readthedocs.io/) represents what I call **embodied intelligence** — analogous to *embodied carbon* in construction. PyWavelets has been tested, debugged, and battle-tested across thousands of real-world scientific applications over two decades. Its C convolution routines encode subtle numerical decisions (boundary handling, filter alignment, periodization conventions) that were arrived at through years of evolutionary pressure from real users discovering real bugs.

This embodied intelligence should not be discarded. But PyWavelets is written in C/Cython and cannot participate in JAX's transformation ecosystem: no automatic differentiation, no JIT compilation, no GPU acceleration, no `jax.vmap` composability. The task of *transferring* this embodied intelligence to JAX — preserving numerical correctness while gaining differentiability and composability — is ideally suited to AI-assisted reference-guided translation.

The key insight is that reimplementing a well-tested library is fundamentally different from writing novel code. The specification already exists in the form of the reference implementation. Every numerical decision can be validated to machine precision. The AI doesn't need domain creativity — it needs **faithful translation and the discipline to match floating-point behaviour exactly**.

## 2. Three Principles

### 2.1 Embodied Intelligence Extraction

PyWavelets' C code contains hundreds of subtle numerical decisions. Rather than reimplementing wavelet transforms from textbook definitions alone, I used PyWavelets as a *numerical oracle*: implement from the mathematical definition, but validate every computation against PyWavelets to machine precision.

This surfaced real issues that textbook implementations would miss:

- **Filter alignment in SWT.** The stationary wavelet transform dilates filters by powers of 2. The naive "dilate, pad, convolve" approach works for Haar but fails for longer filters (db2, db4, db8). Three separate review rounds were needed to identify the correct cycle-spinning alignment matching PyWavelets' [C convolution code](https://github.com/PyWavelets/pywt/blob/main/pywt/_extensions/c/convolution.template.c).
- **Periodization boundary handling.** The periodization mode requires promoting odd-length signals to even, wrap-padding, and careful index alignment. The first implementation produced errors ~1.0; correct behaviour was reached only after studying the C source directly.
- **`sinc` precision in frequency B-spline wavelets.** The `fbsp` wavelet showed `1.5e-14` error vs PyWavelets while all others matched at `~1e-16`. Investigation traced this to `jnp.sinc` vs a manual `sin(πz)/(πz)` implementation. Using `jnp.sinc` (which may be GPU-optimised in future) and accepting `1.5e-14` was a deliberate scientific judgment.
- **Complex arithmetic removal.** PyWavelets' C code computes real and imaginary parts of complex wavelets (Morlet, complex Gaussian) separately. The agent's initial recommendation was "complex numbers are fine on GPU"; I overrode this based on prior student experience with GPU performance and insisted on matching the C code's separated approach. This led to removing all complex arithmetic from the CWT pipeline.

The test suite is 1,177 parametrised cases comparing every function against PyWavelets across multiple wavelet families, signal lengths, decomposition levels, and boundary modes, at tolerances of `atol=1e-14` (direct) and `atol=1e-11` (roundtrip).

### 2.2 Adversarial Iterative Review

The session used **65 external LLM reviews** (GPT-5.4 via [`mcp__llm__review`](https://github.com/handley-lab/mcp-handley-lab)). Of these, **39 returned "FIXES REQUIRED" and 26 returned "APPROVED"** — a 60% rejection rate that reflects genuinely adversarial review, not rubber-stamping.

Reviews operated at multiple levels:

1. **Plan review** (8 iterations) before any implementation, catching incorrect boundary-mode mappings, confusion between `vmap` semantics and structural axis operations, and incorrect convolution conventions.
2. **Implementation review** of each phase against both correctness and the codebase philosophy — a [`CLAUDE.md`](https://github.com/handley-lab/jaxwavelets/blob/master/CLAUDE.md) demanding no defensive programming, no overengineering, no input validation, let-it-crash semantics. This is scientific code, not web services.
3. **Targeted debugging** when implementations produced wrong values: the agent consulted GPT as a *consultant* on specific numerical issues, ultimately fixed by reading PyWavelets' C source directly.
4. **Philosophy enforcement passes** specifically targeting anti-scientific patterns: defensive guards, string-dispatch tables, and unnecessary indirection were caught and removed.

Reviewer-vs-agent disagreement happened. At one point, when the agent appeared to be rationalising a gap in the implementation rather than honestly assessing it, I challenged: *"is it niche, or are you backwards justifying that?"* The agent re-consulted GPT and revised its assessment.

### 2.3 Scientist-as-Architect

I never wrote a line of code. But I made every decision that required scientific judgment.

Of the 95 prompts, the role distribution was roughly:

| Role | Count | Examples |
|------|-------|---------|
| Strategic direction | ~25 | "how much effort to migrate pywt to jax", "review what functionality we're missing" |
| Review triggers | ~20 | "get an iterative review", "aggressive review for anti-scientific programming" |
| Design decisions | ~15 | "I disagree — match the C code", "jaxwavelets — goes with pywavelets" |
| Quality challenges | ~10 | "are you _sure_?", "why 1e-11 and not lower?", "is it niche or backwards justifying?" |
| Operational | ~15 | "commit", "push", "monitor CI" |
| Continuation | ~10 | "continue", "yes", "please don't stop again" |

Specific scientific judgments only the human could make:

- **Tolerance decisions.** Pushing `atol` from `1e-11` to `1e-14`; identifying that the `fbsp` discrepancy was acceptable and explainable.
- **API design.** `import jaxwavelets as wt`, mirroring `import pywt`.
- **Complex arithmetic on GPU.** Overriding the agent's GPU advice based on prior student experience.
- **Philosophy enforcement.** "No defensive programming, let it crash — this is scientific code."
- **Scope control.** "No, let's leave that for now" (on admissible-basis reconstruction).
- **Tone.** The agent's initial README positioned `jaxwavelets` *competitively* against PyWavelets. I recognised that the library *depends on* PyWavelets' embodied intelligence and insisted the framing reflect gratitude rather than competition.

The last point matters. The README change is small, but it is the kind of thing only a person who understood the academic culture of the wavelets community would have done.

## 3. What Required Human Intervention

This list is the credibility check. The AI did not run autonomously. The following decisions required either correcting the agent's first answer, or making a call that no test could validate:

- **Complex arithmetic on GPU.** Agent defaulted to "complex numbers are fine on GPU"; human override based on student experience.
- **Tolerance investigation.** Agent initially set `atol=1e-11` "empirically" without investigating; "why not lower?" forced proper investigation that tightened to `1e-14`.
- **Philosophy enforcement.** Required an explicit `CLAUDE.md` and dedicated review passes; left to itself, the agent drifted toward defensive programming.
- **Naming and framing.** `jaxwavelets` (matching `pywavelets`); collaborative-not-competitive README tone.
- **Scope control.** Knowing when to stop adding features.

## 4. The Workflow

The reference-porting workflow that emerged:

1. Identify a well-tested reference implementation (the embodied intelligence).
2. Write a philosophy document (`CLAUDE.md`) codifying code style and constraints.
3. Plan with adversarial external-LLM review (iterate until the external reviewer approves).
4. Implement phase-by-phase, with:
   1. Implementation,
   2. Test against reference to machine precision,
   3. External review against philosophy,
   4. Human judgment on design decisions and scope.
5. Package, test, publish.
6. Final adversarial review pass for philosophy violations.

The key enabler is step 4(b): an *objective* quality signal. Unlike novel code where "correct" is ambiguous, reimplementation has a clear numerical target. This makes AI-assisted porting particularly effective — the AI handles the translation, the reference implementation provides the specification, the human provides judgment.

## 5. What happened

The session began during a live supervision meeting with PhD student Chris Lovell, who was using PyWavelets for wavelet decomposition of cosmological fields. The question *"how jaxable is pywt?"* led to discovering that the existing JAX wavelet library ([`jax-wavelet-toolbox`](https://github.com/v0lta/Jax-Wavelet-Toolbox)) was unmaintained (last updated 2023). Within 20 minutes we decided to build a fresh implementation.

- Wall-clock span: ~46 hours, Tuesday afternoon to Thursday afternoon.
- Active computation: **4.7 hours** (sum of adjacent-message gaps in the JSONL transcript, excluding any gap >5 minutes — see [Methods note](#methods-note)).
- Human attention: **~25 minutes** spread across 95 prompts.
- Commits: 57.

The agent did not work continuously overnight: after I asked it to continue while I slept, it stalled for **5 hours 22 minutes** then completed an 18-minute autonomous burst at 4am, implementing the continuous wavelet transform and wavelet packets and pushing. Likely cause: a Claude Code idle timeout or session suspension. Detail in the Methods note.

## 6. Reproducibility

- Library: [github.com/handley-lab/jaxwavelets](https://github.com/handley-lab/jaxwavelets) · [pypi.org/project/jaxwavelets](https://pypi.org/project/jaxwavelets/)
- Test suite: 1,177 parametrised cases at `atol=1e-14` (direct) and `atol=1e-11` (roundtrip) against PyWavelets.
- Philosophy document: [`CLAUDE.md`](https://github.com/handley-lab/jaxwavelets/blob/master/CLAUDE.md).
- Filter coefficients (build-time-extracted from PyWavelets): [`scripts/extract_filters.py`](https://github.com/handley-lab/jaxwavelets/blob/master/scripts/extract_filters.py) → [`jaxwavelets/_filters.py`](https://github.com/handley-lab/jaxwavelets/blob/master/jaxwavelets/_filters.py).
- Session JSONL (4,221 entries, 95 prompts, 65 review calls): available on request — please [email](mailto:wh260@cam.ac.uk).

## 7. Limitations

**Maintainability.** The library was generated in a single AI-assisted session. The code is clean, well-tested, and follows a consistent philosophy, but the question of long-term maintainability is open. If a bug surfaces in six months, can a graduate student — or I — debug code we never wrote? The 100% test coverage and machine-precision validation against PyWavelets provide strong regression protection, but the *understanding* of the code resides in the session transcript, not in a human's head. Mitigated by the small library size (~3,000 lines excluding filters) and the `CLAUDE.md` philosophy document.

**Generalisability.** The conditions that made this case successful — a well-tested reference implementation, an objective numerical validation target, and a pure-functional target framework — are not universal. Reimplementing a library with no reference oracle, or building novel scientific software where "correct" is ambiguous, would require a fundamentally different workflow. The embodied-intelligence-extraction pattern applies specifically to the *transfer* of mature software between frameworks.

**Cost.** ~$100 in API spend for a complete library is modest, but would be prohibitive if multiple failed attempts had been needed. This succeeded on the first try, which may not be representative.

**Autonomous operation.** The overnight stall demonstrates that current AI agents cannot reliably sustain multi-hour autonomous work. The productive overnight output was an 18-minute burst, not a sustained 7-hour effort. Autonomous operation remains most effective in short blocks (CI fixing, single-phase implementation) rather than extended campaigns.

## Methods note

The 4.7-hour figure is computed by summing adjacent message-pair gaps in the JSONL transcript, excluding any pair with a gap exceeding 5 minutes. Each adjacent pair falls into one of three categories:

| Category | Time | Count | What it measures |
|----------|------|-------|-----------------|
| Tool execution (assistant→tool result) | 140 min | 1,111 | Wall-clock time for Bash, reviews, file I/O |
| AI thinking → text response | 72 min | 580 | Token generation producing text output |
| AI thinking → tool call | 72 min | 529 | Token generation producing next action |
| **Total active computation** | **284 min ≈ 4.7 h** | | |

Human attention was estimated from the gap distribution between AI text outputs and the next human prompt:

| Percentile | Response time |
|-----------|--------------|
| p10 | 1 second |
| p25 | 12 seconds |
| p50 | 23 seconds |
| p75 | 47 seconds |
| p90 | 3 minutes 38 seconds |

p90 outliers are not deep reading — they are the human on another screen. Capping each prompt at 30 seconds (generous for "yes", "commit", "push", "continue") gives **~22 minutes total**, distributed across 95 prompts over two days.

Tool-call breakdown: 528 Bash invocations, 181 file edits, 130 file reads, 113 file writes, 65 external LLM reviews, 12 web searches, 8 subagent launches.

The overnight stall: after I prompted *"why would you stop. I want you to complete this overnight whilst I sleep. Please don't stop again"* (22:16), the agent spent 9 minutes extending the plan with CWT and wavelet-packet details, wrote one file (`_cwt.py`), and then fell silent for 5h22m. At 04:01 it resumed with the autonomous burst described above. The most likely cause is a Claude Code idle timeout or session suspension — the agent had no pending tool call to keep it active and no human prompt to respond to, so the session was suspended by the harness and later resumed (possibly triggered by a keepalive or reconnection).
