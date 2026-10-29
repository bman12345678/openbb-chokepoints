"""OpenBB router for the chokepoints extension (obb.chokepoints).

Commands here are thin wrappers: fetch data, call the pure functions in
``openbb_chokepoints.core``, and wrap the results in an OBBject.
"""

from openbb_core.app.router import Router

router = Router(prefix="", description="Chokepoint disruption risk analytics.")
