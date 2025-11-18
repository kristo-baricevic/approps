"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";

type TableRow = {
  row_id: number;
  program_name: string;
  program_ai_name: string;
  program_ai_brief: string;
  amount: number | null;
  page: number;
  fy: number | null;
  bbox: unknown;
};

type TableData = {
  table_id: string;
  file_id: string;
  file_name: string;
  table_label: string;
  created_at: string;
  rows: TableRow[];
};

export default function TablePage() {
  const params = useParams();
  const router = useRouter();
  const tableId = (params?.tableId ?? params?.table_id) as string | undefined;

  const [data, setData] = useState<TableData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reparseLoading, setReparseLoading] = useState(false);
  const [renderUrl, setRenderUrl] = useState<string | null>(null);
  const [renderingRowId, setRenderingRowId] = useState<string | null>(null);
  const [renderLoading, setRenderLoading] = useState(false);

  const apiBase = process.env.NEXT_PUBLIC_API_URL;

  useEffect(() => {
    if (!tableId) {
      setError("No tableId in URL");
      setLoading(false);
      return;
    }

    if (!apiBase) {
      setError("NEXT_PUBLIC_API_URL not configured");
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`${apiBase}/tables/${tableId}`)
      .then((r) => {
        if (!r.ok) {
          throw new Error(`status ${r.status}`);
        }
        return r.json();
      })
      .then((json: TableData) => {
        setData(json);
        setLoading(false);
      })
      .catch((e) => {
        setError(`Failed to load table: ${String(e)}`);
        setLoading(false);
      });
  }, [tableId, apiBase]);

  const formatAmount = (val: number | null) => {
    if (val === null || val === undefined) return "—";
    return val.toLocaleString("en-US", { maximumFractionDigits: 0 });
  };

  const handleReparse = async () => {
    if (!data) return;
    if (!apiBase) {
      setError("NEXT_PUBLIC_API_URL not configured");
      return;
    }

    setReparseLoading(true);
    setError(null);
    console.log("data.file_id", data.file_id);
    try {
      const res = await fetch(`${apiBase}/parse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id: data.file_id,
          table_label: data.table_label,
        }),
      });

      if (!res.ok) {
        throw new Error(`status ${res.status}`);
      }

      const out = await res.json(); // { table_id, audit, count }
      const newTableId = out.table_id as string | undefined;
      if (newTableId) {
        router.push(`/tables/${newTableId}`);
      }
    } catch (e) {
      setError(`Failed to reparse table: ${String(e)}`);
    } finally {
      setReparseLoading(false);
    }
  };

  async function viewSource(row: TableRow) {
    if (!row.row_id) return;
    try {
      setRenderLoading(true);
      setRenderingRowId(String(row.row_id));
      setRenderUrl(null);

      const r = await fetch(`${apiBase}/render`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row_id: row.row_id }),
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
    <main className="p-6 space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold mb-1">Table Detail</h1>
          <p className="text-sm text-slate-400">
            Table ID:{" "}
            <span className="font-mono break-all">
              {tableId ?? "(missing)"}
            </span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleReparse}
            disabled={reparseLoading || !data}
            className="rounded-lg border border-slate-700 px-3 py-1 text-sm disabled:opacity-50 hover:bg-slate-200 hover:text-slate-900"
          >
            {reparseLoading ? "Reparsing…" : "Reparse table"}
          </button>

          <Link
            href="/"
            className="rounded-lg border border-slate-700 px-3 py-1 text-sm hover:bg-slate-200 hover:text-slate-900"
          >
            ← Back to Home
          </Link>
        </div>
      </header>

      {loading && <p className="text-sm text-slate-400">Loading…</p>}

      {error && !loading && <p className="text-sm text-red-400">{error}</p>}

      {!loading && !error && data && (
        <>
          <section className="rounded-2xl border border-slate-700 p-4 space-y-1">
            <h2 className="text-lg font-semibold">{data.file_name}</h2>
            <p className="text-sm text-slate-300">
              Subcommittee:{" "}
              <span className="font-mono">{data.table_label}</span>
            </p>
            <p className="text-xs text-slate-500">
              Parsed at: {new Date(data.created_at).toLocaleString()}
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-md font-semibold">Programs</h3>

            {data.rows.length === 0 && (
              <p className="text-sm text-slate-400">
                No rows found for this table.
              </p>
            )}

            {data.rows.length > 0 && (
              <div className="overflow-x-auto rounded-2xl border border-slate-700">
                <table className="min-w-full text-sm">
                  <thead className="bg-slate-900/60">
                    <tr>
                      <th className="px-3 py-2 text-left">Program</th>
                      <th className="px-3 py-2 text-right">Amount ($)</th>
                      <th className="px-3 py-2 text-left">FY</th>
                      <th className="px-3 py-2 text-left">Page</th>
                      <th className="px-3 py-2 text-left">Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((row, idx) => (
                      <tr
                        key={`${row.program_name}-${row.page}-${idx}`}
                        className="border-t border-slate-800 hover:bg-slate-800/40"
                      >
                        <td className="px-3 py-2">
                          <div className="flex flex-col">
                            <div className="flex">
                              {row.program_ai_name
                                ? row.program_ai_name
                                : row.program_name}
                            </div>
                            <div className="flex text-gray-500">
                              {row.program_ai_brief}
                            </div>
                          </div>
                        </td>
                        <td className="px-3 py-2 text-right font-mono">
                          {formatAmount(row.amount)}
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-300">
                          {row.fy ?? "—"}
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-300">
                          {row.page}
                        </td>
                        <td className="p-2 text-center">
                          <button
                            className="text-xs px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 cursor-pointer"
                            onClick={() => viewSource(row)}
                          >
                            View source
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
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
