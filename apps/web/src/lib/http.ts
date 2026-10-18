type ApiErrorBody = {
  code?: string;
  message?: string;
};

type ApiResponse<T> = {
  success: boolean;
  data: T | null;
  error: ApiErrorBody | null;
};

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getApiUrl(path: string): string {
  const baseUrl = "/api/v1";
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;

  return `${baseUrl}${normalizedPath}`;
}

export async function apiRequest<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(getApiUrl(path), options);
  let body: ApiResponse<T> | null = null;

  try {
    body = (await response.json()) as ApiResponse<T>;
  } catch {
    throw new ApiError(
      response.ok ? "API returned an invalid JSON response" : `API request failed with status ${response.status}`,
      response.status,
    );
  }

  if (!response.ok || !body.success || body.data === null) {
    throw new ApiError(
      body.error?.message ?? `API request failed with status ${response.status}`,
      response.status,
      body.error?.code,
    );
  }

  return body.data;
}
