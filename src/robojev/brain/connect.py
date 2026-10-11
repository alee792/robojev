"""Connecting to a robot server: the manifest is checked once, and a non-conforming one is refused.

A standard skill name with another argument shape would be planned, prompted and recorded as the
standard skill while the robot does something else; a catalog version the brain was not written
against may mean exactly that. Both are refused before anything moves (docs/architecture.md,
"Lessons from MCP friction": schema drift). Everything the brain later reads of the manifest
(skill names, arm ids, numbers) has passed through here.
"""
from __future__ import annotations

from .. import catalog
from ..protocol import ArmSpec, Manifest, RobotServer


class ManifestRefused(ValueError):
    """The server's manifest does not conform; the message lists every problem found."""


def problems(m: Manifest) -> list[str]:
    """Every reason to refuse `m`: catalog version, skills that drift from the catalog, no arms,
    duplicate arm ids, skills limited to arms that do not exist."""
    out = []
    if m.catalog != catalog.CATALOG_VERSION:
        out.append(f"catalog version {m.catalog!r}, but this brain speaks {catalog.CATALOG_VERSION!r}")
    out += catalog.check_manifest_skills(m.skills)
    ids = [a.id for a in m.arms]
    if not ids:
        out.append("no arms")
    if len(set(ids)) != len(ids):
        out.append("arm ids are not unique")
    for s in m.skills:
        unknown = [a for a in s.arms if a not in ids]
        if unknown:
            out.append(f"{s.name}: limited to arms that do not exist ({', '.join(unknown)})")
    return out


def connect(server: RobotServer, arm: str | None = None) -> tuple[Manifest, ArmSpec]:
    """Read and check the manifest; -> (manifest, the arm to drive: `arm`, or the first)."""
    m = server.manifest()
    found = problems(m)
    if arm is not None and arm not in {a.id for a in m.arms}:
        found.append(f"no arm {arm!r}")
    if found:
        raise ManifestRefused(f"{m.robot}: refused at connect: " + "; ".join(found))
    spec = next(a for a in m.arms if arm is None or a.id == arm)
    return m, spec
