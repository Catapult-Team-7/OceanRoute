import AnomalyList from "./AnomalyList";
import GlobalStats from "./GlobalStats";
import PointInspector from "./PointInspector";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <GlobalStats />
      <AnomalyList />
      <PointInspector />
    </aside>
  );
}
