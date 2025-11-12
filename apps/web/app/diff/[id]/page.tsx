"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL as string;
if (!API) {
  throw new Error("Missing NEXT_PUBLIC_API_URL");
}

type DiffRow = {
  program_name: string | null;
  prev_amount: number | null;
  curr_amount: number | null;
  delta_abs: number | null;
  delta_pct: number | null;
};

export default function DiffPage() {
  const params = useParams();
  const diffId = params?.id as string;

  const [rows, setRows] = useState<DiffRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "increases" | "cuts">("all");

  useEffect(() => {
    if (!diffId) return;
    setLoading(true);
    fetch(`${API}/diff/${diffId}`)
      .then((r) => r.json())
      .then((data) => setRows(data))
      .finally(() => setLoading(false));
  }, [diffId]);

  const filtered = rows.filter((r) => {
    if (filter === "all") return true;
    if (filter === "increases") return (r.delta_abs ?? 0) > 0;
    if (filter === "cuts") return (r.delta_abs ?? 0) < 0;
    return true;
  });

  return (
    <main className="p-6 max-w-5xl mx-auto space-y-4">
      <h1 className="text-2xl font-semibold mb-2">Differences</h1>

      <div className="flex items-center gap-4 text-sm">
        <span>Filter:</span>
        <select
          className="bg-slate-800 border border-slate-600 rounded px-2 py-1"
          value={filter}
          onChange={(e) => setFilter(e.target.value as any)}
        >
          <option value="all">All changes</option>
          <option value="increases">Increases only</option>
          <option value="cuts">Cuts only</option>
        </select>
      </div>

      {loading && <p className="text-slate-300 text-sm">Loading diff…</p>}

      {!loading && filtered.length === 0 && (
        <p className="text-slate-300 text-sm">No differences found.</p>
      )}

      {!loading && filtered.length > 0 && (
        <table className="w-full text-sm border mt-4">
          <thead className="bg-slate-600">
            <tr>
              <th className="p-2 text-left">Program</th>
              <th className="p-2 text-right">Prev</th>
              <th className="p-2 text-right">Curr</th>
              <th className="p-2 text-right">Δ (abs)</th>
              <th className="p-2 text-right">Δ (%)</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i} className="border-t">
                <td className="p-2">{r.program_name}</td>
                <td className="p-2 text-right">
                  {r.prev_amount?.toLocaleString() ?? "—"}
                </td>
                <td className="p-2 text-right">
                  {r.curr_amount?.toLocaleString() ?? "—"}
                </td>
                <td className="p-2 text-right">
                  {r.delta_abs?.toLocaleString() ?? "—"}
                </td>
                <td className="p-2 text-right">
                  {r.delta_pct != null ? `${r.delta_pct.toFixed(1)}%` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
