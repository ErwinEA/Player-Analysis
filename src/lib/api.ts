import type { PlayerDetails } from "@/components/Sidebar";
import type { AnalyzeResponse } from "@/types/analysis";
import type { InsightsRequest, InsightsResponse } from "@/types/insights";

export type PitchFrameResponse = {
  name: string;
  frame_index: number;
  width: number;
  height: number;
  image_jpeg_base64: string;
  corner_labels: string[];
  boundary_labels?: string[];
  image_corners: number[][] | null;
  image_boundary_points?: number[][] | null;
};

const DEFAULT_API_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;
}

export function getPitchTemplateUrl(name = "testmatch2"): string {
  return `${getApiBaseUrl()}/api/pitch/template?name=${encodeURIComponent(name)}`;
}

export type MobileSamHealth = {
  enabled: boolean;
  import_ok: boolean;
  weights_found: boolean;
  weights_path: string | null;
  loaded: boolean;
  unavailable: boolean;
  unavailable_reason: string | null;
  status: "ok" | "loading" | "error";
};

export type OllamaHealth = {
  reachable: boolean;
  model: string;
  base_url: string;
};

export type HealthResponse = {
  status: string;
  mobile_sam: MobileSamHealth;
  ollama?: OllamaHealth;
};

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${getApiBaseUrl()}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`);
  }
  return res.json() as Promise<HealthResponse>;
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: string | { msg?: string }[] };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d) => d.msg ?? String(d)).join("; ");
    }
  } catch {
    // ignore JSON parse errors
  }
  return `Request failed (${res.status})`;
}

export async function analyzeVideo(
  video: File,
  details: PlayerDetails,
  options?: { calibrationName?: string | null; renderVideo?: boolean },
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("video", video);
  form.append("details", JSON.stringify(details));
  if (options?.calibrationName?.trim()) {
    form.append("calibration_name", options.calibrationName.trim());
  }
  if (options?.renderVideo) {
    form.append("render_video", "true");
  }

  const res = await fetch(`${getApiBaseUrl()}/api/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }

  return res.json() as Promise<AnalyzeResponse>;
}

export async function fetchInsights(
  payload: InsightsRequest,
): Promise<InsightsResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/insights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(await parseErrorDetail(res));
  }

  return res.json() as Promise<InsightsResponse>;
}

export async function fetchPitchFrame(
  name = "testmatch2",
  frameIndex = 100,
): Promise<PitchFrameResponse> {
  const params = new URLSearchParams({
    name,
    frame: String(frameIndex),
  });
  const res = await fetch(`${getApiBaseUrl()}/api/pitch/frame?${params}`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<PitchFrameResponse>;
}

export type PitchCalibrationPreviewResponse = {
  confidence: number;
  coverage_pct: number;
  probe_count: number;
  probe_total: number;
  warnings: string[];
  fitted_quad: number[][];
  mode: string;
  probe_details?: Record<string, unknown>[];
};

export type PitchCalibrationSaveResult = {
  name: string;
  template_url: string;
  calibration_url?: string;
  confidence?: number;
  coverage_pct?: number;
  warnings?: string[];
};

export async function previewPitchCalibration(payload: {
  name: string;
  frame_index: number;
  image_boundary_points: number[][];
  image_width?: number;
  image_height?: number;
}): Promise<PitchCalibrationPreviewResponse> {
  const res = await fetch(`${getApiBaseUrl()}/api/pitch/calibration/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<PitchCalibrationPreviewResponse>;
}

export async function savePitchCalibration(payload: {
  name: string;
  frame_index: number;
  image_boundary_points: number[][];
  image_corners?: number[][];
}): Promise<PitchCalibrationSaveResult> {
  const res = await fetch(`${getApiBaseUrl()}/api/pitch/calibration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<PitchCalibrationSaveResult>;
}

export async function saveLegacyPitchCalibration(
  video: File,
  payload: {
    name: string;
    frame_index: number;
    image_boundary_points: number[][];
  },
): Promise<PitchCalibrationSaveResult> {
  const form = new FormData();
  form.append("name", payload.name);
  form.append("frame_index", String(payload.frame_index));
  form.append(
    "image_corners",
    JSON.stringify(payload.image_boundary_points),
  );
  form.append("video", video);
  let res: Response;
  try {
    res = await fetch(`${getApiBaseUrl()}/api/pitch/calibration/upload`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw new Error(
      "Could not upload video to the server (connection failed or timed out). " +
        "Check that the backend is running and you have enough free disk space.",
    );
  }
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<PitchCalibrationSaveResult>;
}

export async function hasPitchCalibration(name: string): Promise<boolean> {
  const res = await fetch(
    `${getApiBaseUrl()}/api/pitch/calibration?name=${encodeURIComponent(name)}`,
  );
  return res.ok;
}
