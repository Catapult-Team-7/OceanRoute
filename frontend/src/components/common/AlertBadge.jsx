export default function AlertBadge({ severity }) {
  return <span className={`alert-badge ${severity}`}>{severity}</span>;
}
