export function ComingSoon({
  title,
  milestone,
  explanation,
}: {
  title: string;
  milestone: string;
  explanation: string;
}) {
  return (
    <div className="panel">
      <h1>{title}</h1>
      <p className="muted">
        Not built yet — planned for {milestone}. {explanation}
      </p>
    </div>
  );
}
