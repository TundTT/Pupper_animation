"""Fixed-seed, randomized full-sequence audit without partial starts or auto-reset."""

import functools

import jax
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from jax import numpy as jp

from workspace import configs
from workspace.wheel_align_env import HOLD, PupperWheelAlignEnv, wrap


def make_evaluator(config, make_policy, randomization_fn, model_path=None, count=16, seed=901,
                   env_type=PupperWheelAlignEnv):
    audit_config = config.copy_and_resolve_references()
    audit_config.align.initial_completed_max = 0
    if 'direct' in audit_config and 'reference_reset_probability' in audit_config.direct:
        audit_config.direct.reference_reset_probability = 0.0
    audit_config.automatic_commands = True
    env = env_type(audit_config, model_path=model_path)
    wrapped = DomainRandomizationVmapWrapper(
        env,
        functools.partial(
            randomization_fn, rng=jax.random.split(jax.random.PRNGKey(seed), count)
        ),
    )

    @jax.jit
    def rollout(params):
        policy = make_policy(params, deterministic=True)
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(seed + 1), count))

        def advance(carry, _):
            state, alive, rng = carry
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            stepped = wrapped.step(state, action)

            def freeze(old, new):
                mask = alive.reshape(alive.shape + (1,) * (new.ndim - alive.ndim))
                return jp.where(mask, new, old)

            state = jax.tree.map(freeze, state, stepped)
            errors = wrap(
                state.info["target"] - state.pipeline_state.q[:, env._wheel_qpos]
            )
            held = (state.info["phase"] == HOLD) & alive[:, None]
            sample = (
                jp.abs(errors),
                held,
                state.metrics["body_drift_dist"],
                alive,
                state.metrics["active_scrub_speed"],
                state.info["phase"],
                state.info["rotation_verified"],
            )
            alive &= ~state.done.astype(bool)
            return (state, alive, rng), sample

        (state, alive, _), samples = jax.lax.scan(
            advance,
            (state, jp.ones(count, dtype=bool), jax.random.PRNGKey(seed + 2)),
            None,
            length=config.episode_length,
        )
        final_errors = jp.abs(
            wrap(state.info["target"] - state.pipeline_state.q[:, env._wheel_qpos])
        )
        final_speed = jp.abs(state.pipeline_state.qd[:, env._wheel_dof])
        return state.info["completed"], alive, final_errors, final_speed, samples

    def evaluate(params):
        completed, alive, errors, speed, samples = jax.device_get(rollout(params))
        held_errors, held, drift, valid, scrub, phases, verified = samples
        all_completed = np.all(completed, axis=1)
        settled = np.all(
            (errors < config.align.angle_tolerance)
            & (speed < config.align.speed_tolerance),
            axis=1,
        )
        metrics = {
            "audit/episodes": count,
            "audit/survival_fraction": float(np.mean(alive)),
            "audit/all_four_completed_fraction": float(np.mean(all_completed)),
            "audit/all_four_success_fraction": float(
                np.mean(all_completed & alive & settled)
            ),
            "audit/mean_completed_legs": float(np.mean(np.sum(completed, axis=1))),
            "audit/held_samples": int(np.sum(held)),
            "audit/held_error_mean": float(np.mean(held_errors[held]))
            if np.any(held)
            else 0.0,
            "audit/held_error_p95": float(np.percentile(held_errors[held], 95))
            if np.any(held)
            else 0.0,
            "audit/drift_p95": float(np.percentile(drift[valid], 95)),
            "audit/drift_max": float(np.max(drift[valid])),
            "audit/active_scrub_mean": float(np.mean(scrub[valid])),
        }
        metrics.update(
            {
                f"audit/completed_{leg}_fraction": float(np.mean(completed[:, row]))
                for leg, row in configs.FOOT_ROW_BY_LEG.items()
            }
        )
        for leg, row in configs.FOOT_ROW_BY_LEG.items():
            metrics[f"audit/reached_rotation_{leg}_fraction"] = float(
                np.mean(
                    np.any(
                        ((phases[:, :, row] == 2) | verified[:, :, row]) & valid, axis=0
                    )
                )
            )
            metrics[f"audit/verified_rotation_{leg}_fraction"] = float(
                np.mean(np.any(verified[:, :, row] & valid, axis=0))
            )
        return metrics

    return evaluate
