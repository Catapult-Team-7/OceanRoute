import AnomalyList from "./AnomalyList";
import DataDiagnostics from "./DataDiagnostics";
import GlobalStats from "./GlobalStats";
import PointInspector from "./PointInspector";
import PriorityTargets from "./PriorityTargets";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <GlobalStats />
      <PriorityTargets />
      <AnomalyList />
      <DataDiagnostics />
      <PointInspector />
    </aside>
  );
}
