"""Force Field Builder: MD parameters for the current molecule(s).

Assigns OpenFF Sage (SMIRNOFF) parameters with RDKit SMARTS matching, or GAFF
through acpype / openmmforcefields when they are installed, adds charges and
water/ion models, and writes a ``moleditpy-mdff`` JSON hand-off file for the
GROMACS / OpenMM / LAMMPS input generators, plus GROMACS ``.top``/``.itp``,
OpenMM ForceField XML and LAMMPS data files.
"""

import logging

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Force Field Builder"
PLUGIN_VERSION = "0.2.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Builds MD force-field parameters (OpenFF Sage via RDKit, optional GAFF) and charges "
    "for the current molecules, with water/ion models, and writes moleditpy-mdff JSON, "
    "GROMACS, OpenMM XML and LAMMPS data files."
)
PLUGIN_CATEGORY = "Utility"
PLUGIN_TAGS = ["MD", "Force Field"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = [
    "openff-toolkit",
    "openff-nagl",
    "openmmforcefields",
    "acpype",
]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
# Plugin code is LGPL; the bundled OpenFF Sage force field (data/) is CC-BY-4.0.
PLUGIN_LICENSE = "LGPL-3.0-or-later AND CC-BY-4.0"

WINDOW_ID = "force_field_builder_dialog"

_context = None
_dialog_opened = False
# Parameters restored from a project, waiting for the dialog (or the next build).
_restored = {}


def get_default_settings():
    return {
        "force_field": "sage",
        "charge_method": "auto",
        "water_model": "TIP3P",
        "constrain_h": True,
        "cross_check": True,
        "average_equivalent": True,
        "display": "none",
        "lammps_padding": 10.0,
    }


current_settings = get_default_settings()


def _open_dialog(mw=None):
    """Open (or raise) the dialog. Not named ``run`` to avoid a duplicate menu entry."""
    global _dialog_opened

    if _context is not None:
        mw = _context.get_main_window()
        existing = _context.get_window(WINDOW_ID)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return existing

    from .main_dialog import ForceFieldBuilderDialog

    def _get_molecule():
        try:
            if _context is not None:
                return _context.current_molecule
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
            logger.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
        return getattr(mw, "current_mol", None)

    def _mark_modified():
        if _context is not None:
            try:
                _context.mark_project_modified()
            except Exception:
                logger.debug(
                    "%s: mark_project_modified failed", PLUGIN_NAME, exc_info=True
                )

    def _set_molecule(mol):
        # Undoable replacement through the host, as other plugins do it.
        _context.current_molecule = mol
        _context.push_undo_checkpoint()
        _mark_modified()

    _dialog_opened = True
    dlg = ForceFieldBuilderDialog(
        parent=mw,
        context=_context,
        persistent_settings=current_settings,
        get_molecule=_get_molecule,
        mark_modified=_mark_modified,
        set_molecule=_set_molecule if _context is not None else None,
        restored=_restored,
        version=PLUGIN_VERSION,
    )
    if _context is not None:
        _context.register_window(WINDOW_ID, dlg)
    dlg.show()
    return dlg


def initialize(context):
    """Entry point called once when MoleditPy starts."""
    global _context
    _context = context

    context.add_menu_action(
        "Tools/Force Field Builder...", lambda *_args: _open_dialog()
    )

    def save_state():
        if not _dialog_opened:
            return {}
        state = {"settings": dict(current_settings)}
        dlg = context.get_window(WINDOW_ID)
        params = None
        if dlg is not None:
            try:
                params = dlg.project_state()
            except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
                logger.warning("%s: could not save parameters: %s", PLUGIN_NAME, exc)
        elif _restored.get("mdff"):
            params = dict(_restored)
        if params:
            state["parameters"] = params
        return state

    def load_state(data):
        global _dialog_opened
        if not isinstance(data, dict):
            return
        saved = data.get("settings")
        if isinstance(saved, dict):
            current_settings.update(
                {k: v for k, v in saved.items() if k in current_settings}
            )
        params = data.get("parameters")
        _restored.clear()
        if isinstance(params, dict) and params.get("mdff"):
            _restored.update(params)
            _dialog_opened = True
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None:
            try:
                dlg.apply_settings(current_settings)
                if _restored:
                    dlg.restore_project_state(_restored)
            except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
                logger.warning("%s: could not apply loaded state: %s", PLUGIN_NAME, exc)

    def handle_reset():
        global _dialog_opened
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None and dlg.isVisible():
            # An open dialog keeps its state; it notices the new molecule itself.
            dlg.mark_stale()
            return
        current_settings.clear()
        current_settings.update(get_default_settings())
        _restored.clear()
        _dialog_opened = False

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)
