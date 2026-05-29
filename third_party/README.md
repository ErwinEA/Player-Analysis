# Third-party dependencies

## SoccerTrack-v2

Optional reference clone for SoccerNet / SoccerTrack tooling (not wired into the app API).

```bash
# From project root (git repo):
git submodule update --init --recursive third_party/SoccerTrack-v2

# Or shallow clone if submodule not initialized:
git clone --depth 1 https://github.com/AtomScott/SoccerTrack-v2.git third_party/SoccerTrack-v2
```

Current pinned commit (when cloned): see `third_party/SoccerTrack-v2/.git/HEAD` or run:

```bash
git -C third_party/SoccerTrack-v2 rev-parse HEAD
```

Upstream requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) for heavy inference subprocesses.
