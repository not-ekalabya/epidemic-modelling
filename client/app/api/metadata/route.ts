import { NextResponse } from "next/server";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

function getApiBaseUrl(): string {
  return process.env.PREDICTION_API_BASE_URL ?? DEFAULT_API_BASE_URL;
}

export async function GET() {
  const apiResponse = await fetch(`${getApiBaseUrl()}/metadata`, {
    cache: "no-store",
  });

  const contentType = apiResponse.headers.get("content-type") ?? "";
  const data = contentType.includes("application/json")
    ? await apiResponse.json()
    : { detail: await apiResponse.text() };

  return NextResponse.json(data, { status: apiResponse.status });
}
