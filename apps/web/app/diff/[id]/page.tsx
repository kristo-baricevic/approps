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
  curr_row_id: string | null;
};

export default function DiffPage() {
  const params = useParams();
  const diffId = params?.id as string;

  const [rows, setRows] = useState<DiffRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "increases" | "cuts">("all");
  const [renderUrl, setRenderUrl] = useState<string | null>(null);
  const [renderingRowId, setRenderingRowId] = useState<string | null>(null);
  const [renderLoading, setRenderLoading] = useState(false);

  useEffect(() => {
    if (!diffId) return;
    setLoading(true);
    fetch(`${API}/diff/${diffId}`)
      .then((r) => (r.ok ? r.json() : r.text().then((t) => Promise.reject(t))))
      .then((data) => setRows(data))
      .catch((e) => {
        console.error(e);
        setRows([]);
      })
      .finally(() => setLoading(false));
  }, [diffId]);

  useEffect(() => {
    console.log("renderUrl ", renderUrl);
  }, [renderUrl]);

  const filtered = rows.filter((r) => {
    if (filter === "all") return true;
    if (filter === "increases") return (r.delta_abs ?? 0) > 0;
    if (filter === "cuts") return (r.delta_abs ?? 0) < 0;
    return true;
  });

  async function viewSource(row: DiffRow) {
    if (!row.curr_row_id) return;
    try {
      setRenderLoading(true);
      setRenderingRowId(row.curr_row_id);
      setRenderUrl(null);

      const r = await fetch(`${API}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row_id: row.curr_row_id }),
      });

      if (!r.ok) {
        const msg = await r.text();
        throw new Error(msg || "render failed");
      }

      const data = await r.json();
      setRenderUrl(data.url);
    } catch (e) {
      console.error("viewSource failed", e);
    } finally {
      setRenderLoading(false);
    }
  }

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
      <div className="flex gap-4 my-2">
        {/* Export CSV link */}
        <a
          href={`${API}/export/csv?diff_id=${diffId}`}
          className="text-xs px-3 py-2 rounded bg-slate-700 hover:bg-slate-600"
          download={`diff_${diffId}.csv`}
        >
          Export CSV
        </a>
        {/* Print Brief link */}
        <a
          href={`${API}/brief/pdf?diff_id=${diffId}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs px-3 py-2 rounded bg-slate-700 hover:bg-slate-600"
        >
          Print Brief (PDF)
        </a>
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
              <th className="p-2 text-center">Source</th>
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
                <td className="p-2 text-center">
                  <button
                    className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 cursor-pointer"
                    onClick={() => viewSource(r)}
                  >
                    View source
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {renderingRowId && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-slate-900 border-2 border-slate-300 rounded-lg p-4 max-w-3xl w-full max-h-[90vh] flex flex-col">
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-sm font-semibold">Source highlight</h2>
              <button
                className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 cursor-pointer"
                onClick={() => {
                  setRenderingRowId(null);
                  setRenderUrl(null);
                }}
              >
                Close
              </button>
            </div>
            {renderLoading && (
              <p className="text-slate-300 text-xs mb-2">Rendering…</p>
            )}
            {renderUrl && (
              <div className="flex-1 overflow-auto">
                <img
                  src={renderUrl}
                  alt="highlighted source cell"
                  className="max-w-full h-auto mx-auto"
                />
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
