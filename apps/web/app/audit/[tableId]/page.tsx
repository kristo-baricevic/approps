"use client";
import { useEffect, useState } from "react";

export default function AuditPage({ params }: { params: { tableId: string } }) {
  const API = process.env.NEXT_PUBLIC_API_URL!;
  const [rows, setRows] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/tables/${params.tableId}/preview`)
      .then((r) => (r.ok ? r.json() : r.text().then((t) => Promise.reject(t))))
      .then(setRows)
      .catch((e) =>
        setErr(typeof e === "string" ? e : "Failed to load preview")
      );
  }, [API, params.tableId]);

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-xl font-semibold">Audit (stub) — {params.tableId}</h1>
      {err && <p className="text-red-500">{err}</p>}
      {!err && rows.length === 0 && <p>No rows yet.</p>}
      {rows.length > 0 && (
        <table className="w-full text-sm border border-slate-300">
          <thead className="bg-slate-100">
            <tr>
              <th className="p-2 text-left">Program</th>
              <th className="p-2">FY</th>
              <th className="p-2">Amount</th>
              <th className="p-2">Page</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-t">
                <td className="p-2">{r.program_name}</td>
                <td className="p-2 text-center">{r.fy ?? ""}</td>
                <td className="p-2 text-right">{r.amount}</td>
                <td className="p-2 text-center">{r.page}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </main>
  );
}
