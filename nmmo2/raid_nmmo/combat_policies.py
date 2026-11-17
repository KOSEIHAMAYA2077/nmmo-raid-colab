"""The Stage 1 battle controller uses only the jobs chosen during preparation."""
if __package__:
    from .raid_game import ACT_WAIT
else:
    from raid_game import ACT_WAIT
from sweep_checks import make_comp_policy


def by_current_job(env, obs):
    """Run the validated smart-AoE tactics with each player's current job."""
    if obs["in_prep"]:
        return [ACT_WAIT] * len(obs["jobs"])
    # Target and current jobs are identical for every slot, disabling scripted switches.
    return make_comp_policy(obs["jobs"], healer_mode="smart_aoe")(env, obs, 0)
