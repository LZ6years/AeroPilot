import type { Observation } from "../api/types";

export function Evidence({
  observation,
  selected = false,
}: {
  observation?: Observation;
  selected?: boolean;
}) {
  const rgb = observation?.frames.find((frame) => frame.modality === "rgb");
  return (
    <figure className={`evidence ${selected ? "selected" : ""}`}>
      <div className="evidence-image">
        {rgb?.url ? (
          <img src={rgb.url} alt={`模型观测 ${observation?.observation_id}`} />
        ) : (
          <span>观测图像尚未记录</span>
        )}
        {rgb?.width && (
          <small>
            {rgb.width} × {rgb.height}
          </small>
        )}
      </div>
      <figcaption>{observation?.observation_id || "暂无观测"}</figcaption>
    </figure>
  );
}
