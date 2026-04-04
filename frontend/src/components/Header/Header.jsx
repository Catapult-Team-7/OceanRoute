import LiveIndicator from "./LiveIndicator";

export default function Header() {
  return (
    <header className="header">
      <div>
        <p className="eyebrow">Catapult 2026</p>
        <h1>OceanPulse</h1>
        <p className="subtle">AI-assisted carbon sink mapping for an ocean-first climate demo.</p>
      </div>
      <LiveIndicator />
    </header>
  );
}
