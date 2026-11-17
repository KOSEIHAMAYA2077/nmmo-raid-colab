"""Minimal smoke test for NMMO 2.1.2, minigames (RadioRaid / CommTogether), and PyTorch."""
import tempfile
import torch
import nmmo
from nmmo.minigames import RadioRaid, CommTogether


def main() -> None:
    print(f"nmmo={nmmo.__version__} torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = nmmo.config.Default()
        config.set("PATH_MAPS", tmpdir)
        config.set("MAP_N", 2)
        config.set("MAP_FORCE_GENERATION", True)

        env = nmmo.Env(config)

        # 1. RadioRaid (team combat + NPC waves + communication)
        game_raid = RadioRaid(env)
        obs, _ = env.reset(game=game_raid, seed=1)
        first_agent = next(iter(obs))
        comm_shape = obs[first_agent]["Communication"].shape
        for _ in range(5):
            actions = {aid: {"Comm": {"Token": 1}} for aid in obs}
            obs, rewards, terms, truncs, infos = env.step(actions)
        comm_nonzero = int((obs[next(iter(obs))]["Communication"] != 0).sum())
        print(f"RadioRaid ok: alive={len(obs)}, comm_shape={comm_shape}, comm_nonzero={comm_nonzero}")

        # 2. CommTogether (communication + gathering; requires TEAMS before Env init)
        config2 = nmmo.config.Default()
        config2.set("PATH_MAPS", tmpdir)
        config2.set("MAP_N", 2)
        config2.set("TEAMS", game_raid.teams)
        env2 = nmmo.Env(config2)
        game_comm = CommTogether(env2)
        obs2, _ = env2.reset(game=game_comm, seed=2)
        for _ in range(5):
            obs2, rewards, terms, truncs, infos = env2.step({})
        print(f"CommTogether ok: alive={len(obs2)}")

        # 3. Quick PyTorch tensor check on observation
        t = torch.tensor(obs2[next(iter(obs2))]["Tile"], dtype=torch.float32)
        print(f"Torch tensor from obs ok: shape={tuple(t.shape)}")


if __name__ == "__main__":
    main()
