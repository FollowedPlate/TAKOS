from __future__ import annotations

import math
import os
import datetime
from pathlib import Path
from typing import List, Tuple


Point2D = Tuple[float, float]

"""
generates a mujoco xml, based on the design-tool from Open-Spiral-Robots
https://github.com/ZhanchiWang/Open-Spiral-Robots
"""


def _line_segment_intersection(
    a0: type[float, float], a1: type[float, float], b0: type[float, float], b1: type[float, float]
) -> type[float, float] | None:
    ax, ay = a0
    bx, by = a1
    cx, cy = b0
    dx, dy = b1
    r_x = bx - ax
    r_y = by - ay
    s_x = dx - cx
    s_y = dy - cy
    denom = r_x * s_y - r_y * s_x
    if abs(denom) < 1e-12:
        return None
    t = ((cx - ax) * s_y - (cy - ay) * s_x) / denom
    u = ((cx - ax) * r_y - (cy - ay) * r_x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return ax + t * r_x, ay + t * r_y
    return None

def generate_mujoco_xml(
    output_path: str | Path,
    stl_name: str,
    unit_height: float,
    scale: float,
    num_units: int,
    joint_type: str,
    joint_limit_deg: float | None = None,
    robot_length: float | None = None,
    site_points: tuple[float, float, float, float] | None = None,
    cable_mode: int = 2,
) -> None:
    """Generate a MuJoCo XML chain using a single STL mesh."""
    output_path = Path(output_path)
    num_units = max(1, int(num_units))
    joint_type = joint_type.lower().strip()
    if joint_type not in {"hinge", "ball"}:
        joint_type = "hinge"

    # Scale shrinks each unit by 1/scale (per requirement)
    scale_step = 1.0 / max(scale, 1e-9)
    unit_scale = 0.001  # mm -> m for MuJoCo

    current_scale = 1.0
    meshes: list[str] = []
    for i in range(num_units):
        mesh_name = f"baselink_{i+1}"
        meshes.append(
            f'    <mesh name="{mesh_name}" file="{stl_name}" scale="{current_scale * unit_scale:.9f} {current_scale * unit_scale:.9f} {current_scale * unit_scale:.9f}"/>'
        )
        current_scale *= scale_step

    def joint_xml(stiffness: float, damping: float) -> str:
        if joint_type == "hinge":
            if joint_limit_deg is None:
                return (
                    f'<joint type="hinge" axis="1 0 0" stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'
                )
            limit_rad = math.radians(abs(joint_limit_deg))
            return (
                f'<joint type="hinge" axis="1 0 0" range="{-limit_rad:.9f} {limit_rad:.9f}" '
                f'stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'
            )
        return f'<joint type="ball" stiffness="{stiffness:.6f}" damping="{damping:.6f}"/>'

    def build_sites(scale_factor: float, link_index: int) -> list[str]:
        if site_points is None or robot_length is None:
            return []
        x1, y1, x2, y2 = site_points
        z1 = robot_length - x1
        z2 = robot_length - x2
        cos30 = math.cos(math.radians(30.0))
        sin30 = math.sin(math.radians(30.0))
        sites_mm: list[tuple[float, float, float]] = []
        if cable_mode == 3:
            sites_mm = [
                (0.0, y2, z2),
                (0.0, y1, z1),
                (-y2 * cos30, -y2 * sin30, z2),
                (-y1 * cos30, -y1 * sin30, z1),
                (y2 * cos30, -y2 * sin30, z2),
                (y1 * cos30, -y1 * sin30, z1),
            ]
        else:
            sites_mm = [
                (0.0, y2, z2),
                (0.0, y1, z1),
                (0.0, -y2, z2),
                (0.0, -y1, z1),
            ]
        size_m = 0.001 * scale_factor
        sites_xml: list[str] = []
        for idx, (sx, sy, sz) in enumerate(sites_mm, start=1):
            px = sx * scale_factor * unit_scale
            py = sy * scale_factor * unit_scale
            pz = sz * scale_factor * unit_scale
            sites_xml.append(
                f'<site name="s{link_index}_{idx}" pos="{px:.9f} {py:.9f} {pz:.9f}" size="{size_m:.6f} {size_m:.6f} {size_m:.6f}"/>'
            )
        return sites_xml

    # Build nested bodies (kinematic tree)
    current_scale = 1.0

    lines: list[str] = []
    lines.append('<mujoco model="spiral_robot">')
    lines.append('  <compiler angle="radian"/>')
    lines.append('  <option gravity="0 0 0"/>')
    lines.append('  <asset>')
    lines.extend(meshes)
    lines.append(
        '    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>'
    )
    lines.append(
        '    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>'
    )
    lines.append('  </asset>')
    lines.append('  <worldbody>')
    lines.append('    <geom name="ground" type="plane" size="10 10 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>')
    lines.append(
        '    <light pos="1 0.5 0.7" dir="-1 -0.5 -1" directional="true" castshadow="false" diffuse="1 1 1" specular="0.3 0.3 0.3"/>'
    )

    root_z = (robot_length or 0.0) * unit_scale
    lines.append(f'    <body name="link1" pos="0 0 0">')
    lines.append('      <geom type="mesh" mesh="baselink_1"/>')
    for site in build_sites(1.0, 1):
        lines.append(f'      {site}')

    joint_stiffness = 0.04
    joint_damping = 0.02
    for i in range(1, num_units):
        z_offset = unit_height * current_scale * unit_scale
        link_idx = i + 1
        lines.append(f'      <body name="link{link_idx}" pos="0 0 {z_offset:.9f}">')
        lines.append(f'        {joint_xml(joint_stiffness, joint_damping)}')
        lines.append(f'        <geom type="mesh" mesh="baselink_{i+1}"/>')
        scaled_sites = current_scale * scale_step
        for site in build_sites(scaled_sites, link_idx):
            lines.append(f'        {site}')
        current_scale *= scale_step
        joint_stiffness *= scale_step * scale_step
        joint_damping *= scale_step

    for _ in range(num_units - 1):
        lines.append('      </body>')
    lines.append('    </body>')
    lines.append('  </worldbody>')

    # tendons
    tendon_lengths: dict[str, float] = {}
    if num_units > 0 and site_points is not None and robot_length is not None:
        # Precompute site positions per link for length calculation (meters).
        all_sites: dict[int, list[tuple[float, float, float]]] = {}
        current_scale_len = 1.0
        for link_idx in range(1, num_units + 1):
            all_sites[link_idx] = []
            for site in build_sites(current_scale_len, link_idx):
                try:
                    pos_str = site.split('pos="', 1)[1].split('"', 1)[0]
                    sx, sy, sz = (float(v) for v in pos_str.split())
                    all_sites[link_idx].append((sx, sy, sz))
                except Exception:
                    pass
            current_scale_len *= scale_step

        def _tendon_length(indices: list[tuple[int, int]]) -> float:
            pts = []
            for link_i, site_i in indices:
                sites = all_sites.get(link_i, [])
                if site_i - 1 < 0 or site_i - 1 >= len(sites):
                    continue
                pts.append(sites[site_i - 1])
            total = 0.0
            for a, b in zip(pts, pts[1:]):
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                dz = b[2] - a[2]
                total += (dx * dx + dy * dy + dz * dz) ** 0.5
            return total

        if cable_mode == 3:
            tendon_lengths["tendon1"] = _tendon_length([(i, 1) for i in range(1, num_units + 1)] + [(i, 2) for i in range(1, num_units + 1)])
            tendon_lengths["tendon2"] = _tendon_length([(i, 3) for i in range(1, num_units + 1)] + [(i, 4) for i in range(1, num_units + 1)])
            tendon_lengths["tendon3"] = _tendon_length([(i, 5) for i in range(1, num_units + 1)] + [(i, 6) for i in range(1, num_units + 1)])
        else:
            tendon_lengths["tendon1"] = _tendon_length([(i, 1) for i in range(1, num_units + 1)] + [(i, 2) for i in range(1, num_units + 1)])
            tendon_lengths["tendon2"] = _tendon_length([(i, 3) for i in range(1, num_units + 1)] + [(i, 4) for i in range(1, num_units + 1)])

    if num_units > 0:
        lines.append('  <tendon>')
        def _lr(name):
            if robot_length is None:
                return (0.0, 0.0)
            Lm = robot_length * unit_scale
            return (0.4 * Lm, 2.5 * Lm)
        if cable_mode == 3:
            lr1 = _lr('tendon1')
            lr2 = _lr('tendon2')
            lr3 = _lr('tendon3')
            lines.append(f'    <spatial name="tendon1" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{lr1[0]:.9f} {lr1[1]:.9f}">')
            for i in range(1, num_units + 1):
                lines.append(f'      <site site="s{i}_1"/>')
                lines.append(f'      <site site="s{i}_2"/>')
            lines.append('    </spatial>')
            lines.append(f'    <spatial name="tendon2" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{lr2[0]:.9f} {lr2[1]:.9f}">')
            for i in range(1, num_units + 1):
                lines.append(f'      <site site="s{i}_3"/>')
                lines.append(f'      <site site="s{i}_4"/>')
            lines.append('    </spatial>')
            lines.append(f'    <spatial name="tendon3" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{lr3[0]:.9f} {lr3[1]:.9f}">')
            for i in range(1, num_units + 1):
                lines.append(f'      <site site="s{i}_5"/>')
                lines.append(f'      <site site="s{i}_6"/>')
            lines.append('    </spatial>')
        else:
            lr1 = _lr('tendon1')
            lr2 = _lr('tendon2')
            lines.append(f'    <spatial name="tendon1" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{lr1[0]:.9f} {lr1[1]:.9f}">')
            for i in range(1, num_units + 1):
                lines.append(f'      <site site="s{i}_1"/>')
                lines.append(f'      <site site="s{i}_2"/>')
            lines.append('    </spatial>')
            lines.append(f'    <spatial name="tendon2" width="0.0007" frictionloss="0.1" rgba=".95 .3 .3 1" limited="true" range="{lr2[0]:.9f} {lr2[1]:.9f}">')
            for i in range(1, num_units + 1):
                lines.append(f'      <site site="s{i}_3"/>')
                lines.append(f'      <site site="s{i}_4"/>')
            lines.append('    </spatial>')
        lines.append('  </tendon>')

        lines.append('  <actuator>')
        if cable_mode == 3:
            lines.append(f'    <muscle name="act1" tendon="tendon1" ctrllimited="true" ctrlrange="0 1" force="5" lengthrange="{lr1[0]:.9f} {lr1[1]:.9f}"/>')
            lines.append(f'    <muscle name="act2" tendon="tendon2" ctrllimited="true" ctrlrange="0 1" force="5" lengthrange="{lr2[0]:.9f} {lr2[1]:.9f}"/>')
            lines.append(f'    <muscle name="act3" tendon="tendon3" ctrllimited="true" ctrlrange="0 1" force="5" lengthrange="{lr3[0]:.9f} {lr3[1]:.9f}"/>')
        else:
            lines.append(f'    <muscle name="act1" tendon="tendon1" ctrllimited="true" ctrlrange="0 1" force="5" lengthrange="{lr1[0]:.9f} {lr1[1]:.9f}"/>')
            lines.append(f'    <muscle name="act2" tendon="tendon2" ctrllimited="true" ctrlrange="0 1" force="5" lengthrange="{lr2[0]:.9f} {lr2[1]:.9f}"/>')
        lines.append('  </actuator>')
    lines.append('</mujoco>')
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

class Params:
    a: float = 4.95
    b: float = 0.1764
    dtheta_deg: int = 30
    theta_max_pi: float = 6.0
    p: float = 0.5
    elastic_percent: float = 5.0
    elastic_enabled: bool = True
    extrusion: float = 1.0
    cone_angle1: float = 5.0
    cone_angle2: float = 15.0
    tip_hole_pos: float = 50.0
    tip_hole_size: float = 1.4
    base_hole_pos: float = 90.0
    base_hole_size: float = 3.0
    sim_stiffness: float = 0.5
    sim_damping: float = 0.2
    two_cable: bool = True
    cable3_cut_enabled: bool = True
    cable3_cut_pos: float = 25.0
    cable3_cut_size: float = 14.0
    _polys_primary: List[List[Point2D]] = []
    _polys_mirror: List[List[Point2D]] = []
    _robot_length = 0.0

    def _build_frustum_solid(self):
        if self._robot_length <= 1e-6:
            return None
        try:
            import cadquery as cq
        except Exception:
            return None
        tip_pos = (float(self.tip_hole_pos_spin.value()) / 100.0) if hasattr(self,
                                                                             "tip_hole_pos_spin") else self.params.tip_hole_pos
        tip_size = float(self.tip_hole_size_spin.value()) if hasattr(self,
                                                                     "tip_hole_size_spin") else self.params.tip_hole_size
        base_pos = (float(self.base_hole_pos_spin.value()) / 100.0) if hasattr(self,
                                                                               "base_hole_pos_spin") else self.params.base_hole_pos
        base_size = float(self.base_hole_size_spin.value()) if hasattr(self,
                                                                       "base_hole_size_spin") else self.params.base_hole_size
        y0 = max(0.0, min(1.0, tip_pos)) * (self._tip_size * 0.5)
        y1 = max(0.0, min(1.0, base_pos)) * (self._base_size * 0.5)
        p0 = (0.0, y0, 0.0)
        p1 = (self._robot_length, y1, 0.0)
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dz = p1[2] - p0[2]
        length_axis = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length_axis <= 1e-6:
            return None

        # Build frustum around +X axis in XY plane, then revolve around X
        profile = [(0.0, 0.0), (length_axis, 0.0), (length_axis, base_size * 0.5), (0.0, tip_size * 0.5)]
        frustum = (
            cq.Workplane("XY")
            .polyline(profile)
            .close()
            .revolve(360, (0, 0, 0), (1, 0, 0))
        )

        # Rotate frustum so its axis aligns with p0->p1, then translate so (length_axis,0,0) maps to p1
        vx, vy, vz = dx / length_axis, dy / length_axis, dz / length_axis
        ax, ay, az = 0.0, 0.0, 0.0
        dot = max(-1.0, min(1.0, vx))
        angle = math.degrees(math.acos(dot))
        if abs(angle) > 1e-6:
            # axis = cross([1,0,0], v) = (0, -vz, vy)
            ax, ay, az = 0.0, -vz, vy
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm > 1e-9:
                ax, ay, az = ax / norm, ay / norm, az / norm
                frustum = frustum.rotate((0, 0, 0), (ax, ay, az), angle)

        # compute rotated endpoint for (length_axis,0,0)
        def rot_vec(vx0, vy0, vz0):
            if abs(angle) <= 1e-6 or (ax == 0.0 and ay == 0.0 and az == 0.0):
                return (vx0, vy0, vz0)
            theta = math.radians(angle)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            kx, ky, kz = ax, ay, az
            dotp = kx * vx0 + ky * vy0 + kz * vz0
            rx = vx0 * cos_t + (ky * vz0 - kz * vy0) * sin_t + kx * dotp * (1.0 - cos_t)
            ry = vy0 * cos_t + (kz * vx0 - kx * vz0) * sin_t + ky * dotp * (1.0 - cos_t)
            rz = vz0 * cos_t + (kx * vy0 - ky * vx0) * sin_t + kz * dotp * (1.0 - cos_t)
            return (rx, ry, rz)

        end_rot = rot_vec(length_axis, 0.0, 0.0)
        tx = p1[0] - end_rot[0]
        ty = p1[1] - end_rot[1]
        tz = p1[2] - end_rot[2]
        frustum = frustum.translate((tx, ty, tz))
        return frustum

    def _build_cable3_extrude_cut_solid(self):
        if self.params.two_cable or (not self.params.cable3_cut_enabled) or self._robot_length <= 1e-6:
            return None
        try:
            import cadquery as cq
        except Exception:
            return None

        # a1 is Cut Position(%) ratio: controls extrude-cut location.
        a1 = max(0.0, min(0.50, float(self.cable3_cut_pos_spin.value()) / 100.0))
        p0 = (0.0, -(1.0 - a1) * self._tip_size * 0.5)
        p1 = (self._robot_length, -(1.0 - a1) * self._base_size * 0.5)
        p2 = (self._robot_length, -self._base_size)
        p3 = (0.0, -self._tip_size)

        cut_depth = max(0.1, 2.0 * self._base_size)
        base_cut = cq.Workplane("XY").polyline([p0, p1, p2, p3]).close().extrude(cut_depth, both=True)

        cuts = None
        for ang in (0.0, 120.0, 240.0):
            inst = base_cut if ang == 0.0 else base_cut.rotate((0, 0, 0), (1, 0, 0), ang)
            cuts = inst if cuts is None else cuts.union(inst)
        return cuts

    def _build_cable3_cone_cut_solid(self):
        if self.params.two_cable or (not self.params.cable3_cut_enabled) or self._robot_length <= 1e-6:
            return None
        try:
            import cadquery as cq
        except Exception:
            return None

        # a2 is Cut Size(%) ratio: controls cone-cut size.
        a2 = max(0.0, min(0.20, float(self.cable3_cut_size_spin.value()) / 100.0))
        # Placement follows the same centerline from a1 (Cut Position).
        a1 = max(0.0, min(0.50, float(self.cable3_cut_pos_spin.value()) / 100.0))

        # Build pp1/pp2, then extend their line toward negative x to hit y=0 at pp0.
        pp1 = (0.0, a2 * self._tip_size)
        pp2 = (self._robot_length, a2 * self._base_size)
        pp3 = (self._robot_length, 0.0)
        dy = pp2[1] - pp1[1]
        if abs(dy) <= 1e-9:
            pp0_x = -self._robot_length
        else:
            pp0_x = -pp1[1] * self._robot_length / dy
        pp0 = (pp0_x, 0.0)

        # Revolved cutter now uses a triangle profile: pp0-pp2-pp3.
        base_cone = (
            cq.Workplane("XY")
            .polyline([pp0, pp2, pp3])
            .close()
            .revolve(360.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        )

        p0 = (0.0, -(1.0 - a1) * self._tip_size * 0.5, 0.0)
        p1 = (self._robot_length, -(1.0 - a1) * self._base_size * 0.5, 0.0)
        theta_deg = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

        aligned = base_cone.rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), theta_deg)
        x3_rot = self._robot_length * math.cos(math.radians(theta_deg))
        y3_rot = self._robot_length * math.sin(math.radians(theta_deg))
        aligned = aligned.translate((p1[0] - x3_rot, p1[1] - y3_rot, p1[2]))

        cuts = None
        for ang in (0.0, 120.0, 240.0):
            inst = aligned if ang == 0.0 else aligned.rotate((0, 0, 0), (1, 0, 0), ang)
            cuts = inst if cuts is None else cuts.union(inst)
        return cuts

    def _build_cable3_cut_solid(self):
        extrude_cuts = self._build_cable3_extrude_cut_solid()
        cone_cuts = self._build_cable3_cone_cut_solid()
        if extrude_cuts is None:
            return cone_cuts
        if cone_cuts is None:
            return extrude_cuts
        return extrude_cuts.union(cone_cuts)


def export_xml(self) -> None:
        print("exporting xml")
        try:
            import cadquery as cq
        except Exception:
            return
        out_dir = os.path.join(os.path.dirname(__file__), "exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xml_dir = os.path.join(out_dir, f"xml_{ts}")
        os.makedirs(xml_dir, exist_ok=True)
        self._robot_length = max(x for x, _y in self.primary[-1])

        # Build only the rightmost unit without elastic layer

        if self.two_cable:
            thickness = max(0.1, float(self.extrusion_spin.value()))
            solid = None
            right_primary = self._polys_primary[-1]
            right_mirror = self._polys_mirror[-1] if self._polys_mirror else None
            for poly in (right_primary, right_mirror):
                if not poly:
                    continue
                wp = cq.Workplane("XY").polyline(poly).close().extrude(thickness / 2.0, both=True)
                solid = wp if solid is None else solid.union(wp)
            if solid is None:
                return
        else:
            solid = None
            right_primary = self._polys_primary[-1]
            wp = (
                cq.Workplane("XY")
                .polyline(right_primary)
                .close()
                .revolve(360, (0, 0, 0), (1, 0, 0))
            )
            solid = wp if solid is None else solid.union(wp)
            if solid is None:
                return

            # Apply frustum holes for 3-cable unit
            frustum = self._build_frustum_solid()
            if frustum is not None:
                holes = None
                for ang in (0.0, 120.0, 240.0):
                    inst = frustum if ang == 0.0 else frustum.rotate((0, 0, 0), (1, 0, 0), ang)
                    holes = inst if holes is None else holes.union(inst)
                if holes is not None:
                    solid = solid.cut(holes)

            # Apply additional 3-cable manufacturing cuts
            cable3_cut = self._build_cable3_cut_solid()
            if cable3_cut is not None:
                solid = solid.cut(cable3_cut)

        # Transform: translate along x by -robot_length, then rotate about y by 90 deg
        solid = solid.translate((-self._robot_length, 0.0, 0.0))
        solid = solid.rotate((0, 0, 0), (0, 1, 0), 90)

        stl_name = "baselink.stl"
        stl_path = os.path.join(xml_dir, stl_name)
        cq.exporters.export(solid.val(), stl_path)

        xml_path = os.path.join(xml_dir, "robot.xml")
        # compute unit height from rightmost quad (x-axis segment)
        unit_height = 0.0
        if self._polys_primary:
            pts = [p for p in self._polys_primary[-1] if abs(p[1]) < 1e-6]
            if len(pts) >= 2:
                unit_height = abs(pts[1][0] - pts[0][0])
            else:
                unit_height = abs(self._polys_primary[-1][2][0] - self._polys_primary[-1][3][0])
        unit_height = max(1e-6, unit_height)
        # compute cable sites based on last unit
        site_points = None
        if self._polys_primary:
            last_poly = self._polys_primary[-1]
            if len(last_poly) >= 4:
                p0_line = (0.0, self.tip_hole_pos * self._tip_size * 0.5)
                p1_line = (self._robot_length, self.base_hole_pos * self._base_size * 0.5)
                dx = p1_line[0] - p0_line[0]
                dy = p1_line[1] - p0_line[1]
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    dx, dy = 1.0, 0.0
                length = (dx * dx + dy * dy) ** 0.5
                dx /= length
                dy /= length
                L = 1e6
                line_a = (p0_line[0] - dx * L, p0_line[1] - dy * L)
                line_b = (p0_line[0] + dx * L, p0_line[1] + dy * L)
                left_edge = (last_poly[3], last_poly[0])
                right_edge = (last_poly[2], last_poly[1])
                left_hit = _line_segment_intersection(line_a, line_b, left_edge[0], left_edge[1])
                right_hit = _line_segment_intersection(line_a, line_b, right_edge[0], right_edge[1])
                if left_hit and right_hit:
                    x1, y1 = left_hit
                    x2, y2 = right_hit
                    site_points = (x1, y1, x2, y2)
                else:
                    # Fallback: compute y on the line at left/right x positions
                    x1 = left_edge[0][0]
                    x2 = right_edge[0][0]
                    if abs(p1_line[0] - p0_line[0]) < 1e-9:
                        # vertical in x, use end y values
                        y1 = p0_line[1]
                        y2 = p1_line[1]
                    else:
                        slope = (p1_line[1] - p0_line[1]) / (p1_line[0] - p0_line[0])
                        y1 = p0_line[1] + slope * (x1 - p0_line[0])
                        y2 = p0_line[1] + slope * (x2 - p0_line[0])
                    site_points = (x1, y1, x2, y2)
        gamma = math.exp(self.b * math.radians(self.dtheta_deg))
        num_units = max(1, len(self._polys_primary))
        joint_type = "hinge" if self.two_cable else "ball"
        try:
            generate_mujoco_xml(
                xml_path,
                stl_name=stl_name,
                unit_height=unit_height,
                scale=gamma,
                num_units=num_units,
                joint_type=joint_type,
                joint_limit_deg=self.dtheta_deg,
                robot_length=self._robot_length,
                site_points=site_points,
                cable_mode=3 if not self.two_cable else 2,
            )
        except Exception as exc:
            print(f"[Export XML] failed: {exc}")
            return



if __name__ == "__main__":
    temp = Params
    export_xml(temp)