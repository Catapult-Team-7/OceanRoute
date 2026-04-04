import {
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function FluxChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={data}>
        <XAxis dataKey="date" hide />
        <YAxis hide domain={["auto", "auto"]} />
        <Tooltip />
        <ReferenceLine y={0} stroke="#ffffff40" />
        <Line type="monotone" dataKey="flux" stroke="#59c3c3" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
