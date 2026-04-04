export default function InfoHint({ label, description }) {
  return (
    <span className="info-hint" tabIndex={0} aria-label={`${label}: ${description}`}>
      <span className="info-hint-trigger" aria-hidden="true">
        !
      </span>
      <span className="info-hint-tooltip" role="tooltip">
        <strong>{label}</strong>
        <span>{description}</span>
      </span>
    </span>
  );
}
