export default function LoadingOverlay() {
  return (
    <div className="loading-overlay">
      <div className="loading-card">
        <div className="spinner" />
        <span>Refreshing ocean state</span>
      </div>
    </div>
  );
}
