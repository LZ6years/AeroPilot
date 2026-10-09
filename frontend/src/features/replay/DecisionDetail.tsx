import { useEffect, useState } from "react";
import {
  ArrowLeft,
  CheckCircle2,
  Circle,
  CircleDot,
  FileText,
  Link2,
  BookOpen,
} from "lucide-react";
import { api } from "../../api/client";
import type { Decision, Json } from "../../api/types";
import { Evidence } from "../../components/Evidence";

const text = (value: unknown) =>
  value === undefined
    ? "未记录"
    : typeof value === "string"
      ? value
      : JSON.stringify(value);

export function DecisionDetail({
  runId,
  decision,
  onBack,
}: {
  runId: string;
  decision: Decision;
  onBack: () => void;
}) {
  const [view, setView] = useState<Decision>(decision);
  const [audit, setAudit] = useState<Json | null>(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("工具与结果");
  const query = `?instruction_id=${encodeURIComponent(decision.instruction_id || "")}`;
  useEffect(() => {
    let alive = true;
    api<Decision>(`/runs/${runId}/decisions/${decision.decision_id}${query}`)
      .then((v) => {
        if (alive) setView(v);
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [runId, decision, query]);
  const loadAudit = () => {
    if (!audit)
      void api<Json>(
        `/runs/${runId}/decisions/${decision.decision_id}/audit${query}`,
      )
        .then(setAudit)
        .catch((e) => setError(e.message));
  };
  const plan = view.task_state.task_plan as { steps?: Json[] } | undefined;
  const knowledge = view.task_state.loaded_knowledge as Json[] | undefined;
  const label = decision.decision_id.replace(/^decision_0*(\d+)$/, "D$1");
  return (
    <main className="detail-page">
      <header className="detail-title">
        <h1>
          <ArrowLeft size={23} /> 决策详情 · {label}
        </h1>
        <button onClick={onBack}>
          <ArrowLeft size={17} /> 返回工作台
        </button>
      </header>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <div className="detail-layout">
        <section className="panel input-panel">
          <h2>模型所用观测</h2>
          <Evidence observation={view.observation} />
          <h2>当前任务状态</h2>
          <div className="value-box">
            <span className="muted">原始目标</span>
            <p>{view.instruction}</p>
          </div>
          <h3>任务计划</h3>
          {plan?.steps?.length ? (
            <ol className="plan">
              {plan.steps.map((step, i) => (
                <li key={i}>
                  {step.status === "completed" ? (
                    <CheckCircle2 size={19} />
                  ) : step.status === "in_progress" ? (
                    <CircleDot size={19} />
                  ) : (
                    <Circle size={19} />
                  )}
                  <span>
                    {text(step.description || step.title || step.goal)}
                    <small> · {text(step.status)}</small>
                  </span>
                </li>
              ))}
            </ol>
          ) : (
            <p className="muted">本次决策未创建任务计划</p>
          )}
          <h3>已加载知识</h3>
          {knowledge?.length ? (
            knowledge.map((k, i) => (
              <div className="value-box" key={i}>
                <BookOpen size={18} />
                {text(k.name || k.title)}
              </div>
            ))
          ) : (
            <p className="muted">本次决策未加载知识</p>
          )}
          <p className="input-provenance">
            {view.instruction_id} · {view.decision_id}
            <br />
            {view.timestamp}
          </p>
        </section>
        <section className="panel output-panel">
          <div role="tablist" aria-label="决策记录分类" className="detail-tabs">
            {["摘要", "工具与结果", "请求记录"].map((name) => (
              <button
                role="tab"
                key={name}
                aria-selected={tab === name}
                onClick={() => {
                  setTab(name);
                  if (name === "请求记录") loadAudit();
                }}
              >
                {name}
              </button>
            ))}
          </div>
          {tab === "请求记录" ? (
            <div className="request-record">
              <h2>已捕获请求与响应</h2>
              <pre>
                {audit ? JSON.stringify(audit, null, 2) : "正在读取记录…"}
              </pre>
            </div>
          ) : tab === "摘要" ? (
            <div className="request-record">
              <h2>公开意图</h2>
              <p className="value-box">
                {view.public_intent || "未提供公开意图"}
              </p>
              <h3>当时的任务状态</h3>
              <pre>{JSON.stringify(view.task_state, null, 2)}</pre>
              <h3>已记录测量</h3>
              <pre>{JSON.stringify(view.measurements || {}, null, 2)}</pre>
            </div>
          ) : (
            <div className="tools-results">
              <h2>
                公开意图{" "}
                <small>{view.public_intent_source || "未记录来源"}</small>
              </h2>
              <p className="value-box">
                {view.public_intent || "未提供公开意图"}
              </p>
              {view.tool_calls.map((call) => {
                const result = view.tool_results.find(
                  (r) => r.tool_call_id === call.tool_call_id,
                );
                const executed = result?.executed_args as Json | undefined;
                const parameters = [
                  ...new Set([
                    ...Object.keys(call.args),
                    ...Object.keys(executed || {}),
                  ]),
                ];
                const interventions = result?.safety_interventions as
                  unknown[] | undefined;
                return (
                  <section
                    className="call-result"
                    key={call.tool_call_id}
                    id={call.tool_call_id}
                  >
                    <h2>工具调用</h2>
                    <div className="tool-call-box">
                      <div className="call-name">
                        <span>{call.tool_call_id}</span>
                        <strong>{call.tool_name}</strong>
                      </div>
                      <pre>{JSON.stringify(call.args)}</pre>
                    </div>
                    <h2>参数对比</h2>
                    <div className="parameter-table-scroll">
                      <table className="parameter-table">
                        <thead>
                          <tr>
                            <th>参数</th>
                            <th>请求值</th>
                            <th>实际执行值</th>
                          </tr>
                        </thead>
                        <tbody>
                          {parameters.map((key) => (
                            <tr key={key}>
                              <td>{key}</td>
                              <td>{text(call.args[key])}</td>
                              <td>{text(executed?.[key])}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="execution-row">
                      <h2>执行结果</h2>
                      <span
                        className={result?.status === "ok" ? "success" : ""}
                      >
                        {result?.status === "ok" ? (
                          <CheckCircle2 size={21} />
                        ) : (
                          <Circle size={21} />
                        )}
                        {text(result?.status || view.status)}
                      </span>
                      <small>
                        安全干预:{" "}
                        {interventions
                          ? interventions.length === 0
                            ? "无"
                            : `${interventions.length} 项`
                          : "未记录"}
                      </small>
                    </div>
                    <details className="result-record">
                      <summary>实际执行与反馈</summary>
                      <pre>{JSON.stringify(result || {}, null, 2)}</pre>
                    </details>
                  </section>
                );
              })}
              <div className="after-observation">
                <h2>执行后观测</h2>
                <Evidence observation={view.observation_after} />
                <p className="muted">
                  {view.observation_after?.observation_id ||
                    "该次决策未产生新的观测"}
                  <br />
                  {view.observation_after?.timestamp || ""}
                </p>
              </div>
              <div className="evidence-links">
                <h2>证据链接</h2>
                {[view.observation, view.observation_after]
                  .filter(Boolean)
                  .map((o) => (
                    <a
                      key={o!.observation_id}
                      href={o!.frames.find((f) => f.modality === "rgb")?.url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <Link2 size={18} />
                      {o!.observation_id}
                    </a>
                  ))}
              </div>
              <details
                className="raw-audit"
                onToggle={(e) => {
                  if (e.currentTarget.open) loadAudit();
                }}
              >
                <summary>
                  <FileText size={22} />
                  原始记录 · 已捕获请求与响应
                </summary>
                <pre>
                  {audit ? JSON.stringify(audit, null, 2) : "正在读取记录…"}
                </pre>
              </details>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
