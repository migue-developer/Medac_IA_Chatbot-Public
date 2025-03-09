export default async function fetchChat<T = any>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  url: string = "http://127.0.0.1:8000",
  body?: any,
  headers: HeadersInit = {},
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
    body: body ? JSON.stringify(body) : undefined,
    ...options,
  });

  if (!response.ok) {
    throw new Error(`HTTP error! Status: ${response.status}`);
  }

  return response.json();
}
