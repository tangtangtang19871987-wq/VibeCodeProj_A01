import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client.js";

export function HealthBadge() {
  const { data, isLoading } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });

  if (isLoading || !data) {
    return <span className="health-badge health-badge--unknown">checking…</span>;
  }

  return (
    <span
      className={"health-badge " + (data.ok ? "health-badge--ok" : "health-badge--down")}
      title={JSON.stringify(data.checks, null, 2)}
    >
      {data.ok ? "daemon healthy" : "daemon degraded"}
    </span>
  );
}
