export type Json = Record<string, unknown>;
export interface Pose extends Json {
  x?: number;
  y?: number;
  z?: number;
  position?: { x?: number; y?: number; z?: number };
  yaw_deg?: number;
  euler_rad?: { yaw?: number };
}
export interface Preview {
  status: string;
  error?: string;
  width?: number;
  height?: number;
  capture_fps?: number;
  capture_p95_ms?: number;
  age_ms?: number;
  camera_name?: string;
}
export interface Session {
  session_id: string;
  run_id: string;
  map_id: string;
  execution: string;
  phase: string;
  busy: boolean;
  control?: { status: string; desired: string; reason: string };
  instruction: string;
  instruction_id?: string;
  latest_pose: Pose;
  error?: string;
  event_error?: string;
  summary?: Json;
  preview: Preview;
}
export interface MapInfo {
  map_id: string;
  scene_id: string;
  protocol: string;
  connection: string;
  preview_supported: boolean;
}
export interface UiEvent {
  schema_version: "operator_event_v1";
  seq: number;
  kind: string;
  payload: Json;
  instruction_id?: string;
  decision_id?: string;
}
export interface Frame {
  frame_id: string;
  modality: string;
  width?: number;
  height?: number;
  url?: string;
}
export interface Observation {
  observation_id: string;
  timestamp: string;
  source: string;
  pose: Pose;
  frames: Frame[];
}
export interface ToolCall {
  tool_call_id: string;
  tool_name: string;
  args: Json;
}
export interface Decision {
  key: string;
  instruction_id: string;
  decision_id: string;
  timestamp: string;
  seq: number;
  instruction: string;
  public_intent: string;
  public_intent_source?: string;
  status: string;
  observation_id?: string;
  observation_after_id?: string;
  task_state: Json;
  tool_calls: ToolCall[];
  tool_results: Json[];
  observation?: Observation;
  observation_after?: Observation;
  measurements?: Json;
}
export interface Timeline {
  run_id: string;
  name: string;
  decisions: Decision[];
  observations: Observation[];
  trajectory: (Pose & { observation_id: string; timestamp: string })[];
  state: Json;
}
export interface SavedRun {
  run_id: string;
  name: string;
  instruction: string;
  completed: boolean;
  stopped: boolean;
  map_id?: string;
  windows: number;
}
