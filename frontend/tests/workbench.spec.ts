import { test, expect, type Page } from "@playwright/test";

test("replay advances while the timeline refreshes", async ({ page }) => {
  await fixture(page, 2);
  await page.goto("/");
  await page.getByRole("button", { name: "回放", exact: true }).click();
  await page.getByRole("button", { name: "播放回放" }).click();
  await expect(
    page.locator(".decision-list button.selected strong"),
  ).toHaveText("D002");
  await page.getByRole("button", { name: "暂停回放" }).click();
});

async function fixture(page: Page, count = 1) {
  let session = {
    session_id: "test-session",
    run_id: "test-run",
    map_id: "cityenviron_win64",
    execution: "managed",
    phase: "ready",
    busy: false,
    instruction: "",
    latest_pose: {},
    preview: { status: "unavailable" },
    control: { status: "finished", desired: "stopped", reason: "operator" },
  };
  const calls: { path: string; body: any }[] = [];
  const decision = {
    key: "i:D001",
    instruction_id: "i",
    decision_id: "D001",
    timestamp: "2026-10-08T08:00:00Z",
    seq: 1,
    instruction: "飞行测试",
    public_intent: "按观测前进",
    public_intent_source: "provider_content",
    status: "ok",
    observation_id: "obs_0001",
    task_state: {},
    tool_calls: [
      {
        tool_call_id: "move_1",
        tool_name: "uav.move_body",
        args: { vx: 1, vy: 0, vz: 0, duration: 2 },
      },
    ],
    tool_results: [
      {
        tool_call_id: "move_1",
        status: "ok",
        requested_args: { vx: 1 },
        executed_args: { vx: 1 },
      },
    ],
  };
  const timeline = {
    run_id: "test-run",
    name: "测试记录",
    decisions: Array.from({ length: count }, (_, i) => ({
      ...decision,
      key: `i:D00${i + 1}`,
      decision_id: `D00${i + 1}`,
    })),
    observations: [],
    trajectory: [],
    state: {},
  };
  await page.routeWebSocket("**/api/sessions/test-session/ws?**", (ws) =>
    ws.onMessage(() => {}),
  );
  await page.route("http://127.0.0.1:5173/api/**", async (route) => {
    const request = route.request(),
      path = new URL(request.url()).pathname;
    let data: unknown = {};
    if (request.method() === "POST") {
      const body = request.postDataJSON();
      calls.push({ path, body });
      if (path.endsWith("/instructions")) {
        session = {
          ...session,
          busy: true,
          phase: "running",
          instruction: body.instruction,
          control: {
            status: "running",
            desired: "running",
            reason: "operator",
          },
        };
        data = { accepted: true };
      } else if (path.endsWith("/control")) {
        const status =
          body.action === "pause"
            ? "paused"
            : body.action === "resume"
              ? "running"
              : "finished";
        session = {
          ...session,
          control: { status, desired: status, reason: "operator" },
          busy: body.action !== "stop",
          phase: body.action === "stop" ? "ready" : "running",
        };
        data = session.control;
      }
    } else if (path === "/api/maps")
      data = [
        {
          map_id: "cityenviron_win64",
          scene_id: "cityenviron",
          protocol: "airsim",
          connection: "local",
          preview_supported: true,
        },
      ];
    else if (path === "/api/sessions") data = [session];
    else if (path === "/api/sessions/test-session") data = session;
    else if (path === "/api/runs")
      data = [
        {
          run_id: "test-run",
          name: "测试记录",
          instruction: "飞行测试",
          completed: true,
          stopped: false,
          windows: 1,
        },
      ];
    else if (path.endsWith("/timeline")) data = timeline;
    else if (path.endsWith("/audit"))
      data = { model_rounds: [{ canonical_turn: decision }] };
    else if (path.includes("/decisions/")) data = decision;
    await route.fulfill({ json: data });
  });
  return calls;
}

test("task submission and pause/resume/stop send exact operator requests", async ({
  page,
}) => {
  const calls = await fixture(page);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page).toHaveTitle("FlyAgent · 飞行工作台");
  await page.getByRole("textbox", { name: "飞行任务" }).fill("向前飞行两米");
  await page.getByRole("button", { name: "发送任务" }).click();
  await expect(
    page.getByRole("button", { name: "暂停", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "暂停", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "继续", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "继续", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "暂停", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "停止任务" }).click();
  await expect(
    page.getByRole("button", { name: "暂停", exact: true }),
  ).toBeDisabled();
  expect(calls.map((c) => c.body.action).filter(Boolean)).toEqual([
    "pause",
    "resume",
    "stop",
  ]);
  expect(calls[0].body.instruction).toBe("向前飞行两米");
  expect(calls[0].body.request_id).toBeTruthy();
  expect(errors).toEqual([]);
});

test("V3 replay opens captured decision details and audit", async ({
  page,
}) => {
  await fixture(page);
  await page.goto("/");
  await page.getByRole("button", { name: "回放", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "V3 决策回放" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "详细记录" }).click();
  await expect(
    page.getByRole("heading", { name: "决策详情 · D001" }),
  ).toBeVisible();
  await expect(page.getByText("实际执行与反馈", { exact: true })).toBeVisible();
  await page.getByText("原始记录 · 已捕获请求与响应", { exact: true }).click();
  await expect(page.locator(".raw-audit pre")).toContainText("canonical_turn");
  await page.getByRole("button", { name: "返回工作台" }).click();
  await expect(
    page.getByRole("heading", { name: "V3 决策回放" }),
  ).toBeVisible();
});

test("mobile navigation and preview settings remain usable without overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await fixture(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await expect(page.getByRole("heading", { name: "工作台设置" })).toBeVisible();
  await page.getByLabel("宽度 (px)").fill("1280");
  await page.getByRole("button", { name: "恢复默认" }).click();
  await expect(page.getByLabel("宽度 (px)")).toHaveValue("960");
  await page.getByRole("button", { name: "环境", exact: true }).click();
  await expect(page.getByRole("heading", { name: "环境连接" })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
