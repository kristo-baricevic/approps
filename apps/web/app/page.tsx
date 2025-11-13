"use client";
import Link from "next/link";
import React, { useEffect, useState } from "react";

type LatestTable = {
  table_id: string;
  file_name: string;
  table_label: string;
  created_at: string;
};

export default function Page() {
  const [status, setStatus] = useState<string>("checking...");
  const [docs, setDocs] = useState<LatestTable[]>([]);
  const [docsError, setDocsError] = useState<string | null>(null);
  const [docsLoading, setDocsLoading] = useState<boolean>(true);

  const links = [{ href: "/upload", label: "Upload" }];

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`)
      .then((r) => r.json())
      .then((j) => setStatus(j.status ?? "unknown"))
      .catch(() => setStatus("down"));
  }, []);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL;
    if (!base) {
      setDocsError("API URL not configured");
      setDocsLoading(false);
      return;
    }

    fetch(`${base}/tables/latest`)
      .then((r) => {
        if (!r.ok) throw new Error("failed");
        return r.json();
      })
      .then((data: LatestTable[]) => {
        setDocs(data);
        setDocsLoading(false);
      })
      .catch(() => {
        setDocsError("failed to load documents");
        setDocsLoading(false);
      });
  }, []);

  console.log("docs ===> ", docs);

  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold mb-2">Appropriations Parser</h1>
      <p className="mb-6">
        API health: <span className="font-mono">{status}</span>
      </p>

      <div className="rounded-2xl border border-slate-700 p-6 mb-6">
        <nav className="border-b border-black/10">
          <ul className="flex sm:space-x-4 justify-center">
            {links.map((link) => (
              <li
                className="hover:bg-blue-300 hover:text-slate-700 hover:rounded-lg sm:px-4 p-2 text-xs sm:text-base"
                key={link.href}
              >
                <Link href={link.href}>
                  <span className="text-mono">{link.label}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>

      <section className="mt-4">
        <h2 className="text-xl font-semibold mb-3">Latest Budget Documents</h2>

        {docsLoading && <p className="text-sm text-slate-400">Loading…</p>}
        {docsError && (
          <p className="text-sm text-red-400">Error: {docsError}</p>
        )}

        {!docsLoading && !docsError && docs.length === 0 && (
          <p className="text-sm text-slate-400">
            No parsed documents yet. Run the seeding script or upload a PDF.
          </p>
        )}

        {!docsLoading && !docsError && docs.length > 0 && (
          <div className="overflow-x-auto rounded-xl border border-slate-700">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-900/60">
                <tr>
                  <th className="px-4 py-2 text-left">Title</th>
                  <th className="px-4 py-2 text-left">Subcommittee</th>
                  <th className="px-4 py-2 text-left">Parsed At</th>
                  <th className="px-4 py-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((doc) => (
                  <tr
                    key={doc.table_id}
                    className="border-t border-slate-800 hover:bg-slate-800/50"
                  >
                    <td className="px-4 py-2">{doc.file_name}</td>
                    <td className="px-4 py-2">{doc.table_label}</td>
                    <td className="px-4 py-2 text-xs text-slate-400">
                      {new Date(doc.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-2 text-right space-x-2">
                      <Link
                        href={`/tables/${doc.table_id}`}
                        className="inline-block rounded-lg border border-slate-500 px-3 py-1 text-xs hover:bg-slate-200 hover:text-slate-900"
                      >
                        View
                      </Link>
                      {/* Compare will need more wiring later */}
                      <button
                        className="inline-block rounded-lg border border-slate-700 px-3 py-1 text-xs text-slate-400 cursor-not-allowed"
                        title="Compare: coming soon"
                      >
                        Compare
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
