import { useEffect, useRef, useState } from "react";
import {
  Navigation,
  Plane,
  PlaySquare,
  Layers3,
  Settings2,
  Pause,
  Play,
  OctagonX,
  Send,
  FileText,
  CheckCircle2,
  Circle,
  Maximize2,
  Radio,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import { api, post } from "../api/client";
import { useSession } from "../api/useSession";
import type {
  Session,
  MapInfo,
  Timeline,
  Decision,
  SavedRun,
  Pose,
} from "../api/types";
import { Trajectory, scalar } from "../features/flight/Trajectory";
import { DecisionInspector } from "../features/flight/DecisionInspector";
import { DecisionDetail } from "../features/replay/DecisionDetail";

type Page = "flight" | "replay" | "environments" | "settings";
const pages = [
  { id: "flight", label: "飞行", icon: Plane },
  { id: "replay", label: "回放", icon: PlaySquare },
  { id: "environments", label: "环境", icon: Layers3 },
  { id: "settings", label: "设置", icon: Settings2 },
] as const;
const format = (value: unknown, suffix = "") =>
  typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(1)}${suffix}`
    : "—";
const mapLabel = (id: string) =>
  id === "cityenviron_win64" ? "CityEnviron" : id;

export function App() {
  const [page, setPage] = useState<Page>("flight"),
    [sessionId, setSessionId] = useState<string | null>(null);
  const { session, connected, error: sessionError } = useSession(sessionId);
  const [maps, setMaps] = useState<MapInfo[]>([]),
    [mapId, setMapId] = useState("cityenviron_win64");
  const [execution, setExecution] = useState("managed"),
    [error, setError] = useState(""),
    [pending, setPending] = useState(false);
  const [timeline, setTimeline] = useState<Timeline>(),
    [selected, setSelected] = useState<string | null>(null),
    [detail, setDetail] = useState<Decision | null>(null);
  const [runs, setRuns] = useState<SavedRun[]>([]),
    [replayId, setReplayId] = useState<string | null>(null),
    [replayPlaying, setReplayPlaying] = useState(false);
  const [instruction, setInstruction] = useState(""),
    [previewWidth, setPreviewWidth] = useState(960),
    [previewHeight, setPreviewHeight] = useState(540),
    [previewFps, setPreviewFps] = useState(20);
  const decisionList = useRef<HTMLDivElement>(null);
  const active =
    session && !["closed", "failed", "cleanup_failed"].includes(session.phase);
  const runId = page === "replay" ? replayId : session?.run_id;
  const decision =
    timeline?.decisions.find((d) => d.key === selected) ||
    timeline?.decisions.at(-1);

  useEffect(() => {
    let alive = true;
    void Promise.all([api<MapInfo[]>("/maps"), api<Session[]>("/sessions")])
      .then(([m, s]) => {
        if (alive) {
          setMaps(m);
          const current = s.find(
            (v) => !["closed", "failed"].includes(v.phase),
          );
          if (current) {
            setSessionId(current.session_id);
            setMapId(current.map_id);
          }
        }
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, []);
  useEffect(() => {
    if (page !== "replay") return;
    let alive = true;
    api<SavedRun[]>("/runs")
      .then((r) => {
        if (alive) {
          setRuns(r);
          setReplayId((v) => v || r[0]?.run_id || null);
        }
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [page]);
  useEffect(() => {
    setTimeline(undefined);
    setSelected(null);
    setDetail(null);
    setReplayPlaying(false);
    if (!runId) return;
    let alive = true,
      loading = false;
    const refresh = async () => {
      if (loading) return;
      loading = true;
      try {
        const t = await api<Timeline>(`/runs/${runId}/timeline`);
        if (alive) setTimeline(t);
      } catch (e) {
        if (alive) setError((e as Error).message);
      } finally {
        loading = false;
      }
    };
    void refresh();
    const timer = setInterval(refresh, 1000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [runId]);
  useEffect(() => {
    if (!replayPlaying || !timeline) return;
    const timer = setInterval(() => {
      setSelected((current) => {
        const i = timeline.decisions.findIndex((d) => d.key === current);
        if (i >= timeline.decisions.length - 1) {
          setReplayPlaying(false);
          return current;
        }
        return timeline.decisions[i + 1]?.key || null;
      });
    }, 1500);
    return () => clearInterval(timer);
  }, [replayPlaying, timeline?.decisions.length]);
  useEffect(() => {
    if (selected === null && decisionList.current)
      decisionList.current.scrollTop = decisionList.current.scrollHeight;
  }, [timeline?.decisions.length, selected]);

  const action = async (operation: () => Promise<unknown>) => {
    setPending(true);
    setError("");
    try {
      await operation();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  };
  const connect = () =>
    action(async () => {
      const s = await post<Session>("/sessions", {
        map_id: mapId,
        execution,
        preview_width: previewWidth,
        preview_height: previewHeight,
        preview_fps: previewFps,
      });
      setSessionId(s.session_id);
      setPage("flight");
    });
  const signal = (command: "pause" | "resume" | "stop") =>
    action(() => post(`/sessions/${sessionId}/control`, { action: command }));
  const sendTask = () =>
    action(async () => {
      await post(`/sessions/${sessionId}/instructions`, {
        instruction,
        request_id: crypto.randomUUID(),
      });
      setInstruction("");
      setSelected(null);
    });
  const controlStatus = session?.control?.status;
  const pauseText =
    controlStatus === "paused"
      ? "继续"
      : controlStatus === "pause_requested"
        ? "暂停请求中"
        : controlStatus === "resume_requested"
          ? "恢复请求中"
          : "暂停";
  const showDetail = () => {
    if (decision) setDetail(decision);
  };
  const observation =
    timeline?.observations.find(
      (o) => o.observation_id === decision?.observation_id,
    ) || timeline?.observations.at(-1);
  const pose: Pose =
    page === "replay"
      ? observation?.pose || {}
      : session?.busy
        ? session.latest_pose
        : timeline?.observations.at(-1)?.pose || session?.latest_pose || {};
  const velocity = pose.velocity as
    { x?: number; y?: number; z?: number } | undefined;
  const speed =
    velocity &&
    [velocity.x, velocity.y, velocity.z].every((v) => typeof v === "number")
      ? Math.hypot(velocity.x!, velocity.y!, velocity.z!)
      : undefined;
  const trajectory =
    page === "replay"
      ? timeline?.trajectory.filter(
          (p) =>
            timeline.observations.findIndex(
              (o) => o.observation_id === p.observation_id,
            ) <=
            timeline.observations.findIndex(
              (o) => o.observation_id === decision?.observation_id,
            ),
        ) || []
      : timeline?.trajectory || [];
  const yaw =
    pose.yaw_deg ??
    (pose.euler_rad?.yaw === undefined
      ? undefined
      : (pose.euler_rad.yaw * 180) / Math.PI);
  const z = scalar(pose, "z");
  const issue = error || sessionError || session?.error || session?.event_error;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="FlyAgent 首页">
          <Navigation className="brand-mark" size={32} fill="currentColor" />
          <span>FlyAgent</span>
        </a>
        <div className="environment-select">
          <label htmlFor="environment">环境</label>
          <select
            id="environment"
            value={mapId}
            disabled={!!active}
            onChange={(e) => setMapId(e.target.value)}
          >
            {maps.map((m) => (
              <option key={m.map_id} value={m.map_id}>
                {mapLabel(m.map_id)}
              </option>
            ))}
          </select>
        </div>
        <span className={`connection ${connected && active ? "online" : ""}`}>
          <i />
          {connected && active ? "已连接" : active ? "连接中" : "未连接"}
        </span>
        <span className="mode-label">
          当前模式 <strong>自主飞行</strong>
        </span>
        <div className="flight-controls">
          <button
            onClick={() =>
              signal(controlStatus === "paused" ? "resume" : "pause")
            }
            disabled={
              !session?.busy || pending || controlStatus?.endsWith("_requested")
            }
          >
            {controlStatus === "paused" ? (
              <Play size={17} />
            ) : (
              <Pause size={17} />
            )}{" "}
            {pauseText}
          </button>
          <button
            className="danger"
            onClick={() => signal("stop")}
            disabled={
              !session?.busy || pending || controlStatus === "stop_requested"
            }
          >
            <OctagonX size={17} />
            {controlStatus === "stop_requested" ? "停止请求中" : "停止任务"}
          </button>
        </div>
      </header>
      <nav className="sidebar" aria-label="工作台导航">
        {pages.map((p) => (
          <button
            key={p.id}
            className={page === p.id ? "active" : ""}
            aria-current={page === p.id ? "page" : undefined}
            onClick={() => {
              setPage(p.id);
              setDetail(null);
            }}
          >
            <p.icon size={27} />
            <span>{p.label}</span>
          </button>
        ))}
        <div className="sidebar-bottom">
          <Radio size={19} />
          <span>LOCAL</span>
        </div>
      </nav>
      <div className="workspace">
        {issue && (
          <div className="error-banner" role="alert">
            <span>{issue}</span>
            <button aria-label="关闭提示" onClick={() => setError("")}>
              关闭
            </button>
          </div>
        )}
        {session?.busy &&
          session.control?.reason === "control_disconnect" &&
          session.control.desired === "paused" && (
            <div className="pause-banner">
              控制连接中断，已请求暂停。连接恢复后点击“继续”恢复任务。
            </div>
          )}
        {detail && runId ? (
          <DecisionDetail
            runId={runId}
            decision={detail}
            onBack={() => setDetail(null)}
          />
        ) : page === "flight" || page === "replay" ? (
          <>
            {page === "replay" && (
              <div className="replay-toolbar">
                <h1>V3 决策回放</h1>
                <select
                  aria-label="选择飞行记录"
                  value={replayId || ""}
                  onChange={(e) => setReplayId(e.target.value)}
                >
                  <option value="" disabled>
                    选择飞行记录
                  </option>
                  {runs.map((r) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.name} · {r.instruction.slice(0, 40)}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => {
                    if (!replayPlaying) {
                      setSelected(timeline?.decisions[0]?.key || null);
                    }
                    setReplayPlaying((v) => !v);
                  }}
                  disabled={!timeline?.decisions.length}
                >
                  {replayPlaying ? <Pause size={16} /> : <Play size={16} />}{" "}
                  {replayPlaying ? "暂停回放" : "播放回放"}
                </button>
              </div>
            )}
            <main className="flight-grid">
              <div className="flight-main">
                <section className="preview panel">
                  <div className="preview-image">
                    {page === "flight" &&
                    active &&
                    session.preview.status === "streaming" ? (
                      <img
                        src={`/api/sessions/${sessionId}/preview.mjpeg`}
                        alt="无人机独立实时预览"
                      />
                    ) : page === "replay" &&
                      observation?.frames.find((f) => f.modality === "rgb")
                        ?.url ? (
                      <img
                        src={
                          observation.frames.find((f) => f.modality === "rgb")!
                            .url
                        }
                        alt={`回放观测 ${observation.observation_id}`}
                        className="replay-image"
                      />
                    ) : (
                      <div className="preview-empty">
                        <Plane size={52} />
                        <h2>
                          {page === "replay"
                            ? "选择飞行记录开始回放"
                            : session?.phase === "starting"
                              ? "正在启动 CityEnviron 场景"
                              : active
                                ? "等待预览相机"
                                : "准备开始飞行"}
                        </h2>
                        <p>
                          {page === "replay"
                            ? "按决策查看真实观测和执行结果"
                            : active
                              ? session.preview.error ||
                                "连接就绪后将显示实时画面"
                              : "连接 CityEnviron，查看无人机的飞行与决策"}
                        </p>
                        {!active && page === "flight" && (
                          <button
                            className="primary"
                            onClick={connect}
                            disabled={pending}
                          >
                            连接环境 <ArrowRight size={17} />
                          </button>
                        )}
                      </div>
                    )}
                    <div className="preview-label">
                      <strong>
                        {page === "replay"
                          ? "决策观测 · V3 REPLAY"
                          : "实时预览 · UI CAMERA"}
                      </strong>
                      <span>
                        {page === "replay"
                          ? "回放只使用已记录模型观测"
                          : "预览帧不作为决策证据"}
                      </span>
                    </div>
                    {active && page === "flight" && (
                      <div className="preview-stats">
                        {session.preview.width} × {session.preview.height}
                        <span>
                          {format(session.preview.capture_fps, " FPS")}
                        </span>
                        <span>
                          采集 P95{" "}
                          {format(session.preview.capture_p95_ms, " ms")}
                        </span>
                      </div>
                    )}
                    <button
                      className="fullscreen"
                      aria-label="全屏预览"
                      onClick={(e) =>
                        void e.currentTarget.parentElement?.requestFullscreen()
                      }
                    >
                      <Maximize2 size={17} />
                    </button>
                  </div>
                  <div
                    className="telemetry"
                    title="来自最近一次已记录观测；高度为 NED 原点的 −Z"
                  >
                    <div>
                      <span>高度 (NED)</span>
                      <strong>
                        {format(z === undefined ? undefined : -z, " m")}
                      </strong>
                    </div>
                    <div>
                      <span>速度</span>
                      <strong>{format(speed, " m/s")}</strong>
                    </div>
                    <div>
                      <span>航向</span>
                      <strong>{format(yaw, "°")}</strong>
                    </div>
                  </div>
                </section>
                <div className="flight-bottom">
                  <Trajectory points={trajectory} />
                  <section className="panel decision-timeline">
                    <h2>
                      决策时间线{" "}
                      <small>{timeline?.decisions.length || 0} 条记录</small>
                    </h2>
                    <div className="decision-list" ref={decisionList}>
                      {timeline?.decisions.length ? (
                        timeline.decisions.map((d) => (
                          <button
                            key={d.key}
                            className={
                              decision?.key === d.key ? "selected" : ""
                            }
                            onClick={() => setSelected(d.key)}
                          >
                            <span
                              className={`decision-dot ${d.status === "ok" || d.status === "success" ? "success" : ""}`}
                            >
                              {d.status === "pending" ? (
                                <Circle size={19} />
                              ) : d.status === "discarded" ||
                                d.status === "rejected" ? (
                                <OctagonX size={19} />
                              ) : d.status === "ok" ||
                                d.status === "success" ? (
                                <CheckCircle2 size={19} />
                              ) : (
                                <OctagonX size={19} />
                              )}
                            </span>
                            <strong title={d.decision_id}>
                              {d.decision_id.replace(
                                /^decision_0*(\d+)$/,
                                "D$1",
                              )}
                            </strong>
                            <time>
                              {new Date(d.timestamp).toLocaleTimeString(
                                "zh-CN",
                                { hour12: false },
                              )}
                            </time>
                            <span className="decision-intent">
                              {d.public_intent ||
                                d.tool_calls
                                  .map((c) => c.tool_name)
                                  .join(", ") ||
                                "响应记录"}
                            </span>
                          </button>
                        ))
                      ) : (
                        <div className="list-empty">
                          <FileText size={27} />
                          <p>任务开始后，决策将在这里按顺序出现</p>
                        </div>
                      )}
                    </div>
                  </section>
                </div>
                {page === "flight" ? (
                  <form
                    className="command-bar"
                    onSubmit={(e) => {
                      e.preventDefault();
                      if (instruction.trim() && session?.phase === "ready")
                        void sendTask();
                    }}
                  >
                    <FileText size={23} />
                    <input
                      aria-label="飞行任务"
                      placeholder="输入飞行任务…"
                      value={instruction}
                      onChange={(e) => setInstruction(e.target.value)}
                      disabled={!active || session?.busy}
                    />
                    <button
                      type="submit"
                      className="primary"
                      disabled={
                        !instruction.trim() ||
                        session?.phase !== "ready" ||
                        pending
                      }
                    >
                      <Send size={22} />
                      发送任务
                    </button>
                  </form>
                ) : (
                  <div className="replay-footer">
                    <span>仅回放已记录观测，不包含连续预览录像。</span>
                    <button onClick={showDetail} disabled={!decision}>
                      查看决策详情 <ArrowRight size={16} />
                    </button>
                  </div>
                )}
              </div>
              <DecisionInspector
                decision={decision}
                timeline={timeline}
                instruction={session?.instruction || ""}
                onDetail={showDetail}
              />
            </main>
          </>
        ) : page === "environments" ? (
          <main className="configuration-page">
            <h1>环境连接</h1>
            <p className="muted">
              场景由 SimulationServer 管理，飞行与独立预览通过 AirSim RPC 连接。
            </p>
            <section className="panel environment-form">
              <h2>本机场景</h2>
              <label>
                环境
                <select
                  value={mapId}
                  disabled={!!active}
                  onChange={(e) => setMapId(e.target.value)}
                >
                  {maps.map((m) => (
                    <option key={m.map_id} value={m.map_id}>
                      {mapLabel(m.map_id)} · {m.protocol}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                连接方式
                <select
                  value={execution}
                  disabled={!!active}
                  onChange={(e) => setExecution(e.target.value)}
                >
                  <option value="managed">
                    受管启动 · 自动分配端口与场景配置
                  </option>
                  <option value="attach">
                    连接已有场景 · 使用地图配置的 RPC 端口
                  </option>
                </select>
              </label>
              <p className="muted">
                {execution === "managed"
                  ? "创建场景专用 settings 副本，同时配置模型相机与 UI 相机。关闭会话时清理本次启动的场景。"
                  : "已有场景需要预先配置 ui_preview 相机。关闭会话会释放客户端，保留原场景进程。"}
              </p>
              <div className="form-actions">
                <button
                  className="primary"
                  disabled={!!active || pending}
                  onClick={connect}
                >
                  <Radio size={17} />
                  连接环境
                </button>
                <button
                  disabled={!active || pending}
                  onClick={() =>
                    action(() =>
                      api(`/sessions/${sessionId}`, { method: "DELETE" }),
                    )
                  }
                >
                  关闭会话
                </button>
              </div>
              {session && (
                <dl className="session-facts">
                  <dt>场景状态</dt>
                  <dd>{session.phase}</dd>
                  <dt>连接模式</dt>
                  <dd>{session.execution}</dd>
                  <dt>会话</dt>
                  <dd className="code">{session.session_id}</dd>
                  <dt>预览相机</dt>
                  <dd>
                    {session.preview.camera_name || "ui_preview"} ·{" "}
                    {session.preview.status}
                  </dd>
                </dl>
              )}
            </section>
            <section className="panel">
              <h2>连接拓扑</h2>
              <div className="topology">
                <span>React 工作台</span>
                <ArrowRight />
                <span>本机 Python API</span>
                <ArrowRight />
                <span>SimulationServer</span>
                <ArrowRight />
                <span>CityEnviron / AirSim</span>
              </div>
            </section>
          </main>
        ) : (
          <main className="configuration-page">
            <h1>工作台设置</h1>
            <section className="panel environment-form">
              <h2>独立高清预览</h2>
              <p className="muted">
                设置用于下一次场景连接，模型观测分辨率保持地图配置。
              </p>
              <label>
                宽度 (px)
                <input
                  type="number"
                  min="64"
                  max="3840"
                  value={previewWidth}
                  onChange={(e) => setPreviewWidth(Number(e.target.value))}
                />
              </label>
              <label>
                高度 (px)
                <input
                  type="number"
                  min="64"
                  max="2160"
                  value={previewHeight}
                  onChange={(e) => setPreviewHeight(Number(e.target.value))}
                />
              </label>
              <label>
                目标帧率 (FPS)
                <input
                  type="number"
                  min="1"
                  max="60"
                  value={previewFps}
                  onChange={(e) => setPreviewFps(Number(e.target.value))}
                />
              </label>
              <button
                onClick={() => {
                  setPreviewWidth(960);
                  setPreviewHeight(540);
                  setPreviewFps(20);
                }}
              >
                <RefreshCw size={16} />
                恢复默认
              </button>
            </section>
            <section className="panel">
              <h2>执行与连接</h2>
              <dl className="session-facts">
                <dt>断连保护</dt>
                <dd>控制连接断开 5 秒后请求暂停</dd>
                <dt>恢复方式</dt>
                <dd>重连后主动点击“继续”</dd>
                <dt>暂停与停止</dt>
                <dd>在规划或完整工具执行边界生效</dd>
                <dt>决策回放</dt>
                <dd>公开意图、工具调用、执行结果与原始观测</dd>
              </dl>
            </section>
          </main>
        )}
      </div>
    </div>
  );
}
