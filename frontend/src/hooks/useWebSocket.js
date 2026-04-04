import { useEffect, useRef } from "react";

import { WS_URL } from "../utils/constants";
import { useOceanStore } from "../store/oceanStore";

export function useWebSocket() {
  const socketRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const setWsConnected = useOceanStore((state) => state.setWsConnected);

  useEffect(() => {
    let isUnmounted = false;

    function connect() {
      if (isUnmounted) return;
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        reconnectAttemptsRef.current = 0;
        setWsConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload?.type === "ping") {
            return;
          }
        } catch {
          return;
        }
      };

      socket.onerror = () => {
        setWsConnected(false);
        socket.close();
      };

      socket.onclose = () => {
        setWsConnected(false);
        if (isUnmounted) return;
        reconnectAttemptsRef.current += 1;
        const backoffMs = Math.min(20000, 1500 * 2 ** Math.min(reconnectAttemptsRef.current, 4));
        reconnectTimerRef.current = window.setTimeout(connect, backoffMs);
      };
    }

    connect();
    return () => {
      isUnmounted = true;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, [setWsConnected]);
}
