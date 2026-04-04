import { useOceanStore } from "../../store/oceanStore";

export default function LiveIndicator() {
  const wsConnected = useOceanStore((state) => state.wsConnected);

  return (
    <div className="live-indicator">
      <span className={`live-dot ${wsConnected ? "online" : "offline"}`} />
      <span>{wsConnected ? "Live stream connected" : "Live stream offline"}</span>
    </div>
  );
}
