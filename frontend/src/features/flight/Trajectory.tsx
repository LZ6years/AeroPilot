import type { Pose } from "../../api/types";

export function scalar(pose: Pose, axis: "x" | "y" | "z") {
  return pose[axis] ?? pose.position?.[axis];
}
export function Trajectory({ points }: { points: Pose[] }) {
  const valid = points
    .map((p) => [scalar(p, "x"), scalar(p, "y")])
    .filter((p): p is [number, number] =>
      p.every((v) => typeof v === "number" && Number.isFinite(v)),
    );
  const origin = valid[0] || [0, 0];
  const local = valid.map(([x, y]) => [x - origin[0], y - origin[1]]);
  const maxX = Math.max(2, ...local.map((p) => Math.abs(p[0]))),
    maxY = Math.max(2, ...local.map((p) => Math.abs(p[1])));
  const project = ([x, y]: number[]) =>
    `${260 + (x / maxX) * 190},${115 - (y / maxY) * 80}`;
  return (
    <section className="panel trajectory">
      <h2>
        局部轨迹 <small>仅显示已记录位姿 · NED XY</small>
      </h2>
      <svg viewBox="0 0 520 245" role="img" aria-label="已记录的无人机局部轨迹">
        <defs>
          <pattern
            id="grid"
            width="26"
            height="23"
            patternUnits="userSpaceOnUse"
          >
            <path d="M26 0H0V23" fill="none" stroke="#203544" strokeWidth="1" />
          </pattern>
        </defs>
        <rect
          x="40"
          y="18"
          width="455"
          height="185"
          fill="url(#grid)"
          stroke="#263b4a"
        />
        <path d="M40 115H495M260 18V203" stroke="#5b7283" strokeWidth="1" />
        {local.length > 0 && (
          <>
            <polyline
              points={local.map(project).join(" ")}
              fill="none"
              stroke="var(--cyan)"
              strokeWidth="2.5"
            />
            <circle
              cx={project(local.at(-1)!).split(",")[0]}
              cy={project(local.at(-1)!).split(",")[1]}
              r="5"
              fill="var(--cyan)"
            />
            <circle cx="260" cy="115" r="3" fill="var(--text)" />
          </>
        )}
        <text x="262" y="232">
          X (m)
        </text>
        <text x="4" y="118">
          Y (m)
        </text>
        <text x="45" y="218">
          −{maxX.toFixed(1)}
        </text>
        <text x="470" y="218">
          {maxX.toFixed(1)}
        </text>
        <text x="8" y="29">
          {maxY.toFixed(1)}
        </text>
        {local.length === 0 && (
          <text x="260" y="88" textAnchor="middle" className="empty-chart">
            等待飞行位姿记录
          </text>
        )}
      </svg>
    </section>
  );
}
