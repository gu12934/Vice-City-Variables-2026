import { useEffect, useRef, useState } from "react";
import { api, type DataStatus } from "../api";

const mb = (b: number) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;

/**
 * Shown when "Run pipeline" is clicked but the raw competition files aren't on the server.
 * The app itself works without them (results are pre-loaded); they're only needed to recompute.
 */
export function DataUploadDialog({ status, onClose, onReady }: {
  status: DataStatus; onClose: () => void; onReady: () => void;
}) {
  const [picked, setPicked] = useState<Record<string, File>>({});
  const [progress, setProgress] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const uploading = progress !== null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape" && !uploading) onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, uploading]);

  const required = status.files.map((f) => f.name);
  const add = (list: FileList | null) => {
    if (!list) return;
    const next = { ...picked };
    const unknown: string[] = [];
    Array.from(list).forEach((f) => (required.includes(f.name) ? (next[f.name] = f) : unknown.push(f.name)));
    setPicked(next);
    setErr(unknown.length ? `Ignored ${unknown.join(", ")}: the file names must match the ones listed.` : null);
  };

  const missing = status.files.filter((f) => !f.present && !picked[f.name]);
  const toSend = Object.values(picked);
  const canGo = missing.length === 0 && toSend.length > 0 && !uploading;

  const go = async () => {
    setErr(null);
    setProgress(0);
    try {
      const s = await api.uploadData(toSend, setProgress);
      if (!s.ready) throw new Error("Upload finished but some files are still missing.");
      onReady();
    } catch (e) {
      setErr((e as Error).message);
      setProgress(null);
    }
  };

  return (
    <div className="modal-backdrop" onClick={() => !uploading && onClose()}>
      <div className="modal card" role="dialog" aria-modal="true" aria-labelledby="upl-title" onClick={(e) => e.stopPropagation()}>
        <h3 id="upl-title">Add the competition data to run the pipeline</h3>
        <p className="note" style={{ marginTop: 4 }}>
          The results you see are pre-loaded, so nothing is needed to browse the app. To recompute them, provide the three
          parquet files from the competition dataset. They stay on this machine in a Docker volume.
        </p>

        <ul className="file-list">
          {status.files.map((f) => {
            const p = picked[f.name];
            const state = p ? `selected · ${mb(p.size)}` : f.present ? `on server (${f.source}) · ${mb(f.size ?? 0)}` : "missing";
            const ok = !!p || f.present;
            return (
              <li key={f.name}>
                <span className="dot" style={{ background: ok ? "var(--good)" : "var(--axis)" }} />
                <code>{f.name}</code>
                <span className={ok ? "reason-small" : "reason-small warn"}>{state}</span>
              </li>
            );
          })}
        </ul>

        <div
          className={`dropzone ${drag ? "on" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
          onClick={() => input.current?.click()}
          role="button" tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") input.current?.click(); }}
        >
          Drop the .parquet files here, or <u>browse</u>
          <input ref={input} type="file" accept=".parquet" multiple hidden onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
        </div>

        {uploading && (
          <div className="progress-strip" role="status">
            <span>{progress! < 1 ? `Uploading… ${Math.round(progress! * 100)}%` : "Checking files…"}</span>
            <div className="bar"><div style={{ width: `${progress! * 100}%` }} /></div>
          </div>
        )}
        {err && <div className="error-box">{err}</div>}

        <div className="modal-actions">
          <button className="btn" onClick={onClose} disabled={uploading}>Cancel</button>
          <button className="btn primary" onClick={go} disabled={!canGo}>
            {uploading ? "Uploading…" : "Upload & run pipeline"}
          </button>
        </div>
      </div>
    </div>
  );
}
