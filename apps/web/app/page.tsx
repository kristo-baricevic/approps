"use client";
import Link from "next/link";
import React from "react";
import { useEffect, useState } from "react";

export default function Page() {
  const [status, setStatus] = useState<string>("checking...");

  const links = [{ href: "/upload", label: "Upload" }];

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/health`)
      .then((r) => r.json())
      .then((j) => setStatus(j.status ?? "unknown"))
      .catch(() => setStatus("down"));
  }, []);

  return (
    <main className="p-6">
      <h1 className="text-2xl font-semibold mb-2">Appropriations Parser</h1>
      <p className="mb-6">
        API health: <span className="font-mono">{status}</span>
      </p>
      <div className="rounded-2xl border border-slate-700 p-6">
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
    </main>
  );
}
