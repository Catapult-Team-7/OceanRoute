import { useOceanStore } from "../../store/oceanStore";

function getRecentMonths() {
  const items = [];
  const now = new Date();
  for (let i = 5; i >= 0; i -= 1) {
    const date = new Date(now.getFullYear(), now.getMonth() - i, 1);
    items.push(date.toISOString().slice(0, 7));
  }
  return items;
}

export default function TimeSlider() {
  const selectedDate = useOceanStore((state) => state.selectedDate);
  const setSelectedDate = useOceanStore((state) => state.setSelectedDate);
  const months = getRecentMonths();

  return (
    <div className="time-slider">
      <span>Historical view</span>
      <input
        type="range"
        min="0"
        max={String(months.length - 1)}
        value={Math.max(0, months.indexOf(selectedDate))}
        onChange={(event) => setSelectedDate(months[Number(event.target.value)])}
      />
      <strong>{selectedDate}</strong>
    </div>
  );
}
