"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

type TableRow = {
  program_name: string;
  amount: number | null;
  page: number;
  fy: number | null;
  bbox: unknown;
};

type TableData = {
  table_id: string;
  file_name: string;
  table_label: string;
  created_at: string;
  rows: TableRow[];
};

export default function TablePage() {
  const params = useParams();
  const tableId = (params?.tableId ?? params?.table_id) as string | undefined;

  const [data, setData] = useState<TableData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!tableId) {
      setError("No tableId in URL");
      setLoading(false);
      return;
    }

    const base = process.env.NEXT_PUBLIC_API_URL;
    if (!base) {
      setError("NEXT_PUBLIC_API_URL not configured");
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`${base}/tables/${tableId}`)
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
  }, [tableId]);

  const formatAmount = (val: number | null) => {
    if (val === null || val === undefined) return "—";
    return val.toLocaleString("en-US", { maximumFractionDigits: 0 });
  };

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
        <Link
          href="/"
          className="rounded-lg border border-slate-700 px-3 py-1 text-sm hover:bg-slate-200 hover:text-slate-900"
        >
          ← Back to Home
        </Link>
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
                    </tr>
                  </thead>
                  <tbody>
                    {data.rows.map((row, idx) => (
                      <tr
                        key={`${row.program_name}-${row.page}-${idx}`}
                        className="border-t border-slate-800 hover:bg-slate-800/40"
                      >
                        <td className="px-3 py-2">{row.program_name}</td>
                        <td className="px-3 py-2 text-right font-mono">
                          {formatAmount(row.amount)}
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-300">
                          {row.fy ?? "—"}
                        </td>
                        <td className="px-3 py-2 text-xs text-slate-300">
                          {row.page}
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
    </main>
  );
}
