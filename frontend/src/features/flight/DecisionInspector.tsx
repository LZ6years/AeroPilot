import { CheckCircle2, Circle, ChevronRight } from "lucide-react";
import type { Decision, Timeline, Json } from "../../api/types";
import { Evidence } from "../../components/Evidence";

export function toolText(name: string, args: Json) {
  return `${name}(${Object.entries(args)
    .map(
      ([k, v]) =>
        `${k}=${typeof v === "string" ? JSON.stringify(v) : JSON.stringify(v)}`,
    )
    .join(", ")})`;
}
export function DecisionInspector({
  decision,
  timeline,
  instruction,
  onDetail,
}: {
  decision?: Decision;
  timeline?: Timeline;
  instruction: string;
  onDetail: () => void;
}) {
  const observations = timeline?.observations || [];
  const index = observations.findIndex(
    (o) => o.observation_id === decision?.observation_id,
  );
  const plan = decision?.task_state?.task_plan as
    { steps?: Json[] } | undefined;
  return (
    <aside className="inspector">
      <h1>决策流程</h1>
      <div className="inspector-tabs">
        <span className="active">当前决策</span>
        <button onClick={onDetail} disabled={!decision}>
          详细记录 <ChevronRight size={14} />
        </button>
      </div>
      <section>
        <h2>任务目标</h2>
        <div className="value-box">
          {decision?.instruction || instruction || "等待下达飞行任务"}
        </div>
      </section>
      <section>
        <h2>任务计划</h2>
        {plan?.steps?.length ? (
          <ol className="plan">
            {plan.steps.map((step, i) => (
              <li key={i}>
                {step.status === "completed" ? (
                  <CheckCircle2 size={18} />
                ) : (
                  <Circle size={18} />
                )}
                <span>
                  {String(
                    step.description ||
                      step.title ||
                      step.goal ||
                      JSON.stringify(step),
                  )}
                </span>
              </li>
            ))}
          </ol>
        ) : (
          <p className="muted">
            {decision ? "本次决策未创建任务计划" : "任务开始后显示已记录计划"}
          </p>
        )}
      </section>
      <section>
        <h2>
          模型输出{" "}
          <small>
            公开意图 ·{" "}
            {decision?.public_intent_source === "tool_arguments_fallback"
              ? "工具参数摘要"
              : "Provider"}
          </small>
        </h2>
        <div className="value-box">
          {decision?.public_intent ||
            (decision ? "该次响应未提供公开意图" : "等待模型响应")}
        </div>
      </section>
      <section>
        <h2>工具调用</h2>
        {decision?.tool_calls.length ? (
          decision.tool_calls.map((call) => (
            <div
              className="value-box code compact-call"
              key={call.tool_call_id}
              title={toolText(call.tool_name, call.args)}
            >
              {toolText(call.tool_name, call.args).length > 160
                ? `${call.tool_name}(…)`
                : toolText(call.tool_name, call.args)}
            </div>
          ))
        ) : (
          <div className="value-box muted">暂无工具调用</div>
        )}
      </section>
      {decision?.tool_calls.length ? (
        <section>
          <h2>请求参数</h2>
          <div className="request-parameters">
            {decision.tool_calls.flatMap((call) =>
              Object.entries(call.args).map(([key, value]) => (
                <span
                  key={`${call.tool_call_id}:${key}`}
                  title={JSON.stringify(value)}
                >
                  {key} = {JSON.stringify(value)}
                </span>
              )),
            )}
          </div>
        </section>
      ) : null}
      <section>
        <h2>执行结果</h2>
        <p
          className={`result ${decision?.status === "ok" || decision?.status === "success" ? "success" : ""}`}
        >
          {decision?.status === "ok" || decision?.status === "success" ? (
            <CheckCircle2 size={19} />
          ) : (
            <Circle size={19} />
          )}
          {decision?.status || "等待执行"}
        </p>
      </section>
      <section>
        <h2>决策所用观测</h2>
        <div className="evidence-pair">
          <Evidence
            observation={index > 0 ? observations[index - 1] : undefined}
          />
          <Evidence
            observation={index >= 0 ? observations[index] : undefined}
            selected
          />
        </div>
      </section>
      <section>
        <h2>证据引用</h2>
        <p className="muted">
          {decision?.observation_id
            ? `基于 ${decision.observation_id} 进行决策。`
            : "暂无决策观测引用"}
        </p>
        <button className="audit-link" onClick={onDetail} disabled={!decision}>
          原始记录 <ChevronRight size={16} />
        </button>
      </section>
    </aside>
  );
}
