#!/usr/bin/env python3
"""Prepare/apply Inkling 2 Mini prefill optimization on DS4 commit 3393871.

Usage:
  python3 apply_inkling2_prefill_optimization.py /path/to/ds4 --check
  python3 apply_inkling2_prefill_optimization.py /path/to/ds4 --diff > inkling2-prefill.patch
  python3 apply_inkling2_prefill_optimization.py /path/to/ds4 --apply

No network required. Refuses to modify files whose Git blob hashes do not
match the pinned feature branch, and validates every edit before writing.
"""
import argparse
import difflib
import hashlib
import os
from pathlib import Path
import sys
import tempfile

EXPECTED_BLOBS = {
    "ds4.c": "8ad28c7214ec6d5d05e64bcc84e037d7e4b71408",
    "tests/test_inkling2_session.c": "a1002afdbf94de68dd049539960cbf5d56432a9a",
    "docs/INKLING2_METAL.md": "eaf69b29afc5d279256070cf4fda2e36f7277841",
}


def git_blob_sha(data):
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact occurrence, got {count}")
    return text.replace(old, new, 1)


LOGITS_HELPERS = r'''/* The last row's residual mix remains resident in mix_output_rows[0].
 * This lets an interrupted prompt publish a valid checkpoint without
 * rerunning any of its 48 visits. */
static bool inkling_gpu_project_final_logits(
        ds4_session *session, const char **failed_stage) {
    ds4_engine *engine = session->engine;
    ds4_inkling_gpu_context *context = &session->inkling_gpu;
    if (failed_stage) *failed_stage = "output normalization";
    if (!ds4_gpu_rms_norm_weight_tensor(
            context->layer.next, context->layer.mix_output_rows[0],
            engine->model.map, engine->model.size,
            engine->inkling_weights.output_norm->abs_offset,
            INKLING_GPU_HIDDEN, g_ds4_inkling_model_config.rms_epsilon))
        return false;
    if (failed_stage) *failed_stage = "output projection";
    if (!ds4_gpu_matmul_q8_0_tensor(
            context->logits, engine->model.map, engine->model.size,
            engine->inkling_weights.output->abs_offset,
            INKLING_GPU_HIDDEN,
            (uint32_t)engine->inkling_weights.output->dim[1],
            context->layer.next, 1u))
        return false;
    if (failed_stage) *failed_stage = "logit softcap";
    return ds4_gpu_inkling2_logit_softcap_tensor(
            context->logits, context->logits,
            (uint32_t)engine->inkling_weights.output->dim[1],
            g_ds4_inkling_model_config.final_logit_softcap) != 0;
}

/* A cancelled prefix is a public, reusable checkpoint. Materialize its
 * logits on demand from the saved final residual mix before returning. */
static bool inkling_gpu_restore_prefill_logits(
        ds4_session *session, char *error, size_t error_capacity) {
    if (!ds4_gpu_begin_commands()) {
        snprintf(error, error_capacity,
                 "Inkling 2 Mini checkpoint logits could not begin a Metal batch");
        return false;
    }
    const char *failed_stage = "Metal command completion";
    const bool projected = inkling_gpu_project_final_logits(
        session, &failed_stage);
    const bool completed = ds4_gpu_end_commands() != 0;
    if (!projected || !completed) {
        snprintf(error, error_capacity,
                 "Inkling 2 Mini checkpoint logits failed at %s",
                 projected ? "Metal command completion" : failed_stage);
        return false;
    }
    return inkling_gpu_read_logits(session, error, error_capacity);
}

'''

PREFILL_FUNCTION = r'''static int inkling_gpu_prefill(ds4_session *session,
                               const ds4_tokens *prompt, int start,
                               char *error, size_t error_capacity) {
    /* The normal prefill needs output logits only for the final group.
     * Keep the old every-group policy available for exact A/B comparisons. */
    const char *logits_env = getenv("DS4_INKLING_PREFILL_LOGITS_EVERY_GROUP");
    const bool every_group_logits = logits_env && logits_env[0] &&
                                    strcmp(logits_env, "0") != 0;
    const char *profile_env = getenv("DS4_INKLING_PREFILL_PROFILE");
    const bool profile = profile_env && profile_env[0] &&
                         strcmp(profile_env, "0") != 0;
    const double started = profile ? now_sec() : 0.0;
    double queue_ms = 0.0, finish_ms = 0.0, read_ms = 0.0;
    uint32_t groups = 0, logits_groups = 0;
    bool logits_pending = false;

    /* Process each logical visit across a small ordered row group. KV and
     * convolution histories advance in token order while MoE uses row batches. */
    for (int group_start = start; group_start < prompt->len;) {
        if (ds4_session_cancelled(session)) {
            /* The latest group is committed, but an interrupted checkpoint
             * must expose the same logits as a standalone prompt prefix. */
            if (logits_pending) {
                const double t0 = profile ? now_sec() : 0.0;
                if (!inkling_gpu_restore_prefill_logits(
                        session, error, error_capacity)) {
                    ds4_session_invalidate(session);
                    return 1;
                }
                if (profile) read_ms += (now_sec() - t0) * 1000.0;
                logits_groups++;
            }
            if (profile) {
                fprintf(stderr, "ds4: Inkling prefill interrupted tokens=%d groups=%u "
                        "logits_groups=%u elapsed_ms=%.3f queue_ms=%.3f "
                        "finish_ms=%.3f read_and_restore_ms=%.3f\n",
                        group_start - start, groups, logits_groups,
                        (now_sec() - started) * 1000.0, queue_ms, finish_ms, read_ms);
            }
            snprintf(error, error_capacity, "interrupted");
            session->checkpoint_valid = session->checkpoint.len > 0;
            return DS4_SESSION_SYNC_INTERRUPTED;
        }

        session->checkpoint_valid = false;
        int group_end = group_start + INKLING_GPU_PREFILL_BATCH_TOKENS;
        if (group_end > prompt->len) group_end = prompt->len;
        if (!ds4_gpu_begin_commands()) {
            snprintf(error, error_capacity,
                     "Inkling 2 Mini Metal prefill could not begin a command batch");
            ds4_session_invalidate(session);
            return 1;
        }

        const uint32_t group_tokens = (uint32_t)(group_end - group_start);
        const bool want_logits = every_group_logits || group_end == prompt->len;
        const double queue_start = profile ? now_sec() : 0.0;
        bool ok = inkling_gpu_eval_group(session, prompt->v + group_start,
                group_tokens, want_logits, INKLING_GPU_COMMANDS_SHARED,
                error, error_capacity);
        if (profile) queue_ms += (now_sec() - queue_start) * 1000.0;
        if (!ok) {
            (void)ds4_gpu_end_commands();
            ds4_session_invalidate(session);
            return 1;
        }
        const double finish_start = profile ? now_sec() : 0.0;
        if (!ds4_gpu_end_commands()) {
            snprintf(error, error_capacity,
                     "Inkling 2 Mini Metal prefill command batch failed");
            ds4_session_invalidate(session);
            return 1;
        }
        if (profile) finish_ms += (now_sec() - finish_start) * 1000.0;
        if (want_logits) {
            const double read_start = profile ? now_sec() : 0.0;
            if (!inkling_gpu_read_logits(session, error, error_capacity)) {
                ds4_session_invalidate(session);
                return 1;
            }
            if (profile) read_ms += (now_sec() - read_start) * 1000.0;
            logits_groups++;
        }
        logits_pending = !want_logits;

        for (int i = group_start; i < group_end; i++) {
            token_vec_push(&session->checkpoint, prompt->v[i]);
            session->mtp_draft_valid = false;
        }
        session->inkling_gpu.position = (uint32_t)group_end;
        session->checkpoint_valid = true;

        group_start = group_end;
        groups++;
        if (session->progress)
            session->progress(session->progress_ud, "prefill_chunk",
                              group_end, prompt->len);
    }
    if (profile) {
        fprintf(stderr, "ds4: Inkling prefill complete tokens=%d groups=%u "
                "logits_groups=%u skipped_logits=%u elapsed_ms=%.3f "
                "queue_ms=%.3f finish_ms=%.3f read_ms=%.3f "
                "(wall times; finish includes GPU work)\n",
                prompt->len - start, groups, logits_groups,
                groups - logits_groups, (now_sec() - started) * 1000.0,
                queue_ms, finish_ms, read_ms);
    }
    return 0;
}
'''

TEST_ADDITION = '''    /* Compare the historical per-group vocabulary projection with the
     * deferred final-only path, then exercise deferred checkpoint recovery. */
    assert(setenv("DS4_INKLING_PREFILL_LOGITS_EVERY_GROUP", "1", 1) == 0);
    assert(ds4_session_layer_slice_reset(session, error, sizeof(error)) == 0);
    assert(ds4_session_sync(session, &long_prompt, error, sizeof(error)) == 0);
    assert(memcmp(tokenwise_logits, session->logits,
                  (size_t)TEST_VOCAB * sizeof(float)) == 0);
    assert(unsetenv("DS4_INKLING_PREFILL_LOGITS_EVERY_GROUP") == 0);

    assert(ds4_session_layer_slice_reset(session, error, sizeof(error)) == 0);
    prefill_cancel_probe cancel_probe = {0};
'''

DOC_ADDITION = '''## Deferred logits and Q8_0 A/B instrumentation — candidate, M4 validation pending

The native Inkling visit-major prefill evaluates groups of at most eight rows.
Normal prefill now performs the full vocabulary projection and host logit
readback **only for the last group**, rather than each intermediate group.
The last row's final residual mix stays resident in Metal. If cancellation
arrives after an intermediate group, DS4 projects and reads that group's logits
before returning an interrupted, reusable checkpoint. The decoder and the
logical-visit, per-row causal attention/shortconv execution are unchanged.

Two optional environment switches provide controlled, same-prompt comparisons:

```sh
DS4_INKLING_PREFILL_PROFILE=1 <normal DS4 command>
DS4_INKLING_PREFILL_PROFILE=1 DS4_INKLING_PREFILL_LOGITS_EVERY_GROUP=1 <same command>
DS4_INKLING_PREFILL_PROFILE=1 DS4_METAL_Q8_MV_EXT_MAX_TOKENS=2 <same command>
```

The first is the new default (final-group logits), the second restores the
previous every-group behavior, and the third tests the existing Metal
Q8_0 kernel dispatch with the few-row `mul_mv_ext` threshold reduced from
16 to 2; with eight rows it selects the generic matrix path instead. This
third mode **does not** increase Inkling's batch to 32/64 and **does not**
guarantee a faster kernel. Use identical prompts, warmups, settings and the
same M4. Repeat each measurement and compare prefill tokens/s, elapsed time,
generated IDs, memory and resumed-prefix logits.

`DS4_INKLING_PREFILL_PROFILE=1` prints group counts, vocabulary-projection
counts, total wall time, CPU queue/submission time, GPU command-completion
wait and host logit-readback time. These are *wall-clock sections*, not an
independent per-kernel GPU profiler; `queue_ms` may include blocking
backend calls. Avoid enabling the existing `DS4_METAL_Q8_PREFILL_PROFILE`
for throughput comparisons: it forces synchronization around individual
projections and is not representative of normal execution.

`make test-inkling2-session` now checks the old every-group path against
the final-only path and keeps the interrupted-prefix byte-identical logit
comparison. This is a synthetic Metal fixture; real GGUF correctness,
interrupt/resume, actual kernel selection, and stable M4 performance
are still unverified. Keep the split-K attention PR separate.

'''


def patch_source(original):
    text = original
    text = replace_once(
        text,
        'static bool inkling_gpu_eval_group(ds4_session *session, const int *tokens,\n',
        LOGITS_HELPERS + 'static bool inkling_gpu_eval_group(ds4_session *session, const int *tokens,\n',
        'insert logit helpers',
    )
    start = '    if (ok && want_logits) failed_stage = "output normalization";\n'
    end = '    const bool commands_ok = commands_started &&\n'
    i, j = text.find(start), text.find(end, text.find(start))
    if i < 0 or j < 0 or j - i > 2500:
        raise RuntimeError('output projection block not as expected')
    text = (text[:i] +
            '    if (ok && want_logits)\n'
            '        ok = inkling_gpu_project_final_logits(session, &failed_stage);\n' +
            text[j:])
    beginning = 'static int inkling_gpu_prefill(ds4_session *session,\n'
    ending = '\n#undef INKLING_GPU_REQUIRE'
    i, j = text.find(beginning), text.find(ending, text.find(beginning))
    if i < 0 or j < 0 or j - i > 6500:
        raise RuntimeError('old prefill block not as expected')
    text = text[:i] + PREFILL_FUNCTION + text[j:]
    if 'group_tokens, true, INKLING_GPU_COMMANDS_SHARED' in text:
        raise RuntimeError('old every-group unconditional logits call remains')
    return text


def patch_test(original):
    return replace_once(original,
        '    assert(ds4_session_layer_slice_reset(session, error, sizeof(error)) == 0);\n'
        '    prefill_cancel_probe cancel_probe = {0};\n',
        TEST_ADDITION,
        'insert synthetic A/B test')


def patch_docs(original):
    return replace_once(original,
        '## Conversational API and function tools — 2026-11-23',
        DOC_ADDITION + '## Conversational API and function tools — 2026-11-23',
        'insert validation instructions')


PATCHERS = {
    'ds4.c': patch_source,
    'tests/test_inkling2_session.c': patch_test,
    'docs/INKLING2_METAL.md': patch_docs,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('repo', type=Path, help='DS4 repository directory at the pinned feature commit')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='verify pinned source and patch anchors without writing')
    mode.add_argument('--diff', action='store_true', help='print unified git-apply-compatible diff to stdout')
    mode.add_argument('--apply', action='store_true', help='replace each file atomically after validating all three')
    args = parser.parse_args()
    changes = []
    for relative, expected_sha in EXPECTED_BLOBS.items():
        path = args.repo / relative
        original_bytes = path.read_bytes()
        actual_sha = git_blob_sha(original_bytes)
        if actual_sha != expected_sha:
            raise RuntimeError(f'{relative}: expected Git blob {expected_sha}, found {actual_sha}; '
                               'checkout feature/inkling2-mini-metal-m4 at 3393871; no files changed')
        original = original_bytes.decode('utf-8')
        replacement = PATCHERS[relative](original)
        if replacement == original:
            raise RuntimeError(f'{relative}: patch made no changes')
        changes.append((relative, path, original, replacement))
    if args.check:
        for relative, _, before, after in changes:
            print(f'OK {relative}: {len(before)} -> {len(after)} characters', file=sys.stderr)
        print('All source SHAs and replacements verified; no files changed.', file=sys.stderr)
    elif args.diff:
        for relative, _, before, after in changes:
            sys.stdout.writelines(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile='a/' + relative, tofile='b/' + relative, n=3))
    else:
        staged = []
        try:
            for relative, path, _, new in changes:
                with tempfile.NamedTemporaryFile(mode='wb', prefix='.ds4-prefill-',
                                                 dir=path.parent, delete=False) as tmp:
                    tmp.write(new.encode('utf-8'))
                    tmp_path = Path(tmp.name)
                os.chmod(tmp_path, path.stat().st_mode)
                staged.append((tmp_path, path))
            for tmp_path, path in staged:
                os.replace(tmp_path, path)
                print(f'Updated {path}')
        finally:
            for tmp_path, _ in staged:
                tmp_path.unlink(missing_ok=True)
        print('Next: make -B ds4 ds4-server && make test-inkling2-session test-inkling2-metal')
        print('M4 runtime + same-GGUF parity/throughput validation required before merge.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
