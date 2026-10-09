import { useEffect, useRef, useState } from "react";
import { api } from "./client";
import type { Session, UiEvent } from "./types";

/** Events reconnect from the last delivered sequence; reconnection never resumes flight. */
export function useSession(sessionId: string | null) {
  const [session, setSession] = useState<Session | null>(null);
  const [connected, setConnected] = useState(false);
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState("");
  const cursor = useRef(0);
  useEffect(() => {
    cursor.current = 0;
    setSession(null);
    setError("");
    setConnected(false);
    if (!sessionId) return;
    let alive = true,
      socket: WebSocket | null = null;
    let reconnect: ReturnType<typeof setTimeout> | undefined;
    const refresh = async () => {
      try {
        const value = await api<Session>(`/sessions/${sessionId}`);
        if (alive) {
          setSession(value);
          setError("");
        }
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    };
    const connect = () => {
      if (!alive) return;
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/sessions/${sessionId}/ws?after=${cursor.current}`,
      );
      socket.onopen = () => {
        if (alive) {
          setConnected(true);
          socket?.send(JSON.stringify({ type: "ping" }));
          void refresh();
        }
      };
      socket.onmessage = (message) => {
        const event: UiEvent = JSON.parse(message.data);
        if (event.seq > cursor.current) {
          cursor.current = event.seq;
          setRevision((r) => r + 1);
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (alive) reconnect = setTimeout(connect, 1000);
      };
    };
    connect();
    void refresh();
    const timer = setInterval(() => {
      void refresh();
      if (socket?.readyState === WebSocket.OPEN)
        socket.send(JSON.stringify({ type: "ping" }));
    }, 1000);
    return () => {
      alive = false;
      clearInterval(timer);
      clearTimeout(reconnect);
      if (socket) {
        socket.onclose = null;
        socket.close();
      }
    };
  }, [sessionId]);
  return { session, connected, revision, error };
}
