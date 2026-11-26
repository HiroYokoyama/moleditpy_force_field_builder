"""Metadata, registration and project persistence (no Qt, no RDKit needed)."""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import force_field_builder as plugin


class FakeContext:
    def __init__(self):
        self.menu_actions = []
        self.save_handlers = []
        self.load_handlers = []
        self.reset_handlers = []
        self.windows = {}

    def add_menu_action(self, path, callback):
        self.menu_actions.append((path, callback))

    def register_save_handler(self, callback):
        self.save_handlers.append(callback)

    def register_load_handler(self, callback):
        self.load_handlers.append(callback)

    def register_document_reset_handler(self, callback):
        self.reset_handlers.append(callback)

    def get_window(self, window_id):
        return self.windows.get(window_id)

    def get_main_window(self):
        return None


@pytest.fixture
def ctx():
    plugin._dialog_opened = False
    plugin._restored.clear()
    plugin.current_settings.clear()
    plugin.current_settings.update(plugin.get_default_settings())
    context = FakeContext()
    plugin.initialize(context)
    return context


def test_metadata():
    assert plugin.PLUGIN_NAME == "Force Field Builder"
    assert re.fullmatch(r"\d+\.\d+\.\d+", plugin.PLUGIN_VERSION)
    assert plugin.PLUGIN_VERSION == "0.1.0"
    assert plugin.PLUGIN_AUTHOR == "HiroYokoyama"
    assert plugin.PLUGIN_LICENSE == "LGPL-3.0-or-later"
    assert plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION == ">=5.0.0, <6.0.0"
    assert plugin.PLUGIN_CATEGORY == "Utility"
    assert plugin.PLUGIN_DEPENDENCIES == []
    assert not hasattr(plugin, "run"), (
        "a module-level run() would add a duplicate menu entry"
    )


def test_registers_one_menu_action_and_handlers(ctx):
    assert [p for p, _ in ctx.menu_actions] == ["Tools/Force Field Builder..."]
    assert (
        len(ctx.save_handlers) == len(ctx.load_handlers) == len(ctx.reset_handlers) == 1
    )


def test_save_is_empty_until_used(ctx):
    assert ctx.save_handlers[0]() == {}


def test_load_restores_settings_and_parameters(ctx):
    params = {"mdff": {"format": "moleditpy-mdff"}, "fingerprint": [1, "C"]}
    ctx.load_handlers[0](
        {"settings": {"water_model": "SPC/E", "unknown": 1}, "parameters": params}
    )
    assert plugin.current_settings["water_model"] == "SPC/E"
    assert "unknown" not in plugin.current_settings
    assert plugin._restored["fingerprint"] == [1, "C"]
    saved = ctx.save_handlers[0]()
    assert saved["parameters"]["mdff"] == params["mdff"]
    assert saved["settings"]["water_model"] == "SPC/E"


def test_load_ignores_garbage(ctx):
    ctx.load_handlers[0]("not a dict")
    ctx.load_handlers[0]({"parameters": "nope"})
    assert plugin._restored == {}


def test_reset_clears_state_when_closed(ctx):
    ctx.load_handlers[0](
        {"settings": {"water_model": "OPC3"}, "parameters": {"mdff": {"x": 1}}}
    )
    ctx.reset_handlers[0]()
    assert plugin.current_settings == plugin.get_default_settings()
    assert plugin._restored == {}
    assert ctx.save_handlers[0]() == {}


def test_reset_leaves_an_open_dialog_alone(ctx):
    class Dlg:
        stale = False

        def isVisible(self):
            return True

        def mark_stale(self):
            self.stale = True

    dlg = Dlg()
    ctx.windows[plugin.WINDOW_ID] = dlg
    plugin.current_settings["water_model"] = "OPC3"
    ctx.reset_handlers[0]()
    assert dlg.stale
    assert plugin.current_settings["water_model"] == "OPC3"


def test_bundled_force_field_and_license_present():
    data = os.path.join(os.path.dirname(plugin.__file__), "data")
    assert os.path.isfile(os.path.join(data, "openff-2.3.0.offxml"))
    with open(
        os.path.join(data, "LICENSE-openff-forcefields.txt"), encoding="utf-8"
    ) as handle:
        assert "Attribution 4.0 International" in handle.read()


def test_charge_colours_follow_the_esp_convention():
    from force_field_builder import display

    assert display.charge_colour(-1.0) == "#ff0000"
    assert display.charge_colour(1.0) == "#0000ff"
    assert display.charge_colour(0.0) == "#ffffff"
    assert display.charge_colours([-0.5, 0.25])[0] == "#ff0000"
