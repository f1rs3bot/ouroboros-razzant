import importlib.util, json, sys, time, types
from pathlib import Path


def _load_plugin():
    root = Path(__file__).resolve().parents[1] / "skills" / "telegram"
    pkg = types.ModuleType("tg_ht_test"); pkg.__path__ = [str(root)]
    sys.modules["tg_ht_test"] = pkg
    spec = importlib.util.spec_from_file_location("tg_ht_test.plugin", root / "plugin.py")
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m); return m


def _api_with_data(tmp_path):
    data = tmp_path / "data"
    sd = data / "state" / "skills" / "telegram-bridge"
    sd.mkdir(parents=True)
    (data / "logs").mkdir(parents=True, exist_ok=True)

    class A:
        def get_state_dir(self): return str(sd)
        def get_settings(self, keys): return {}
        def log(self, *a, **k): pass
    return A(), data


def test_health_snapshot(tmp_path):
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    (data / "state" / "queue_snapshot.json").write_text(
        json.dumps({"running_count": 2, "pending_count": 3, "running": [], "pending": []}), encoding="utf-8")
    (data / "state" / "worker_pids.json").write_text(
        json.dumps({"workers": [{"pid": 1}, {"pid": 2}, {"pid": 3}]}), encoding="utf-8")
    txt = plugin._collect_health(api, "en")
    assert "RUNNING 2" in txt and "PENDING 3" in txt
    assert "Workers: 3" in txt
    assert "clean" in txt          # no supervisor.jsonl → no incidents
    assert "Disk" in txt


def test_health_degrades_when_files_missing(tmp_path):
    plugin = _load_plugin()
    api, data = _api_with_data(tmp_path)
    # no queue/worker files at all → must not raise, queue shows idle
    txt = plugin._collect_health(api, "ru")
    assert "RUNNING 0" in txt and "PENDING 0" in txt


def test_health_discloses_when_incident_tail_omits_prefix(tmp_path, monkeypatch):
    plugin = _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]
    api, data = _api_with_data(tmp_path)
    monkeypatch.setattr(health, "_SUPERVISOR_TAIL_BYTES", 120)
    log = data / "logs" / "supervisor.jsonl"
    rows = [
        {"ts": "2020-01-01T00:00:00Z", "type": "unrelated", "padding": "x" * 80},
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "type": "unrelated"},
    ]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    txt = plugin._collect_health(api, "en")

    assert "clean" not in txt
    assert "none in bounded tail" in txt
    assert "earlier coverage may be incomplete" in txt
    ru = plugin._collect_health(api, "ru")
    assert "чисто" not in ru
    assert "в ограниченном хвосте не найдено" in ru
    assert "ранняя часть периода может быть неполной" in ru


def test_tasks_idle_and_list(tmp_path):
    _load_plugin()
    health = sys.modules["tg_ht_test.lib.telegram_health"]  # _collect_tasks_text lives here post-split
    api, data = _api_with_data(tmp_path)
    (data / "state" / "queue_snapshot.json").write_text(json.dumps({"running": [], "pending": []}), encoding="utf-8")
    assert "idle" in health._collect_tasks_text(api, "en")
    (data / "state" / "queue_snapshot.json").write_text(json.dumps({
        "running": [{"id": "abc12345", "type": "evolution", "delegation_role": "root"}],
        "pending": [{"id": "def67890", "type": "task", "delegation_role": "subagent"}],
    }), encoding="utf-8")
    txt = health._collect_tasks_text(api, "en")
    assert "evolution" in txt and "abc12345" in txt
    assert "task" in txt and "subagent" in txt


def test_tasks_panel_builds():
    plugin = _load_plugin()

    class A:
        def get_state_dir(self): return "/tmp/nope-telegram-bridge"
    header, kb = plugin._build_menu_tasks(A(), "safe_commands", "ru")
    assert "Задачи" in header
    assert kb[-1][0]["callback_data"] == "nav:menu"




def test_default_command_mode_is_full():
    src = (Path(__file__).resolve().parents[1] / "skills" / "telegram" / "plugin.py").read_text(encoding="utf-8")
    # the command-mode default fallback is full_access (not strict)
    assert 'TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_FULL' in src
    assert 'TELEGRAM_COMMAND_MODE") or _COMMAND_MODE_STRICT' not in src
