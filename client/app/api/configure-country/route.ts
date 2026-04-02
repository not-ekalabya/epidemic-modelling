import { NextRequest, NextResponse } from "next/server";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

function getApiBaseUrl(): string {
  return process.env.PREDICTION_API_BASE_URL ?? DEFAULT_API_BASE_URL;
}

export async function POST(request: NextRequest) {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Request body must be valid JSON." },
      { status: 400 },
    );
  }

  const apiResponse = await fetch(`${getApiBaseUrl()}/config/country`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  const contentType = apiResponse.headers.get("content-type") ?? "";
  const data = contentType.includes("application/json")
    ? await apiResponse.json()
    : { detail: await apiResponse.text() };

  return NextResponse.json(data, { status: apiResponse.status });
}
